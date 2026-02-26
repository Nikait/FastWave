import os
from os import path
import random
import numpy as np
import torch
import librosa as rosa

from glob import glob
from torch.utils.data import Dataset, DataLoader
from src.utils.init_utils import set_worker_seed
from scipy.signal import sosfiltfilt, cheby1, resample_poly


import torch
from typing import List, Dict, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    wavs = [sample["data_object"] for sample in batch]
    wav_ls = [sample["wav_l"] for sample in batch]
    bands = [sample["band"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]

    max_len = max(w.shape[0] for w in wavs)
    padded = lambda lst: [
        torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in lst
    ]
    batch_wavs = torch.stack(padded(wavs), dim=0)
    batch_wav_ls = torch.stack(padded(wav_ls), dim=0)
    batch_bands = torch.stack(bands, dim=0)

    return {
        "data_object": batch_wavs,
        "wav_l": batch_wav_ls,
        "band": batch_bands,
        "filename": filenames
    }


class VCTKMultiSpkDataset(Dataset):
    def __init__(self, hparams, cv=0, sr=16000):  # cv: 0=train, 1=val, 2=test
        self.hparams = hparams
        self.cv = cv
        self.cv_ratio = eval(hparams.datasets.data.cv_ratio)
        self.sr = sr
        self.directory = hparams.datasets.data.dir
        self.dataformat = hparams.datasets.data.format

        self.data_list = self._get_datalist(
            self.directory,
            self.dataformat,
            self._get_speakers(self.directory),
            cv
        )
        
        assert len(self.data_list) > 0, f"No audio data found for cv={cv}!"

    def _get_speakers(self, folder):
        return sorted(glob(os.path.join(folder, '*')))

    def _get_datalist(self, folder, file_format, spk_list, cv):
        _dl = []
        len_spk_list = len(spk_list)
        s = 0
        print(f'full speakers {len_spk_list}')
        for i, spk in enumerate(spk_list):
            if cv == 0:
                if not (i < int(len_spk_list * self.cv_ratio[0])): continue
            elif cv == 1:
                if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                        i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                    continue
            else:
                if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                        i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                    continue
            _full_spk_dl = sorted(glob(path.join(spk, file_format)))
            _len = len(_full_spk_dl)
            if (_len == 0): continue
            s += 1
            _dl.extend(_full_spk_dl)

        return _dl
    

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav, _ = rosa.load(path=self.data_list[index], sr=self.hparams.audio.sampling_rate)
        wav /= np.max(np.abs(wav))

        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv < 2 else padl // 2
            wav = np.pad(wav, (r, padl - r), 'constant', constant_values=0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start + self.hparams.audio.length] if self.cv < 2 \
                else wav[:len(wav) - len(wav) % self.hparams.audio.hop_length]
        wav *= random.random() / 2 + 0.5 if self.cv < 2 else 1

        if self.cv == 0:
            order = random.randint(1, 11)
            ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
            highcut = random.randint(self.hparams.audio.sr_min // 2, self.hparams.audio.sr_max // 2)
        else:
            order = 8
            ripple = 0.05
            if self.cv == 1:
                highcut = random.choice([8000 // 2, 12000 // 2, 16000 // 2, 24000 // 2])
            elif self.cv == 2:
                highcut = self.sr // 2

        nyq = 0.5 * self.hparams.audio.sampling_rate
        hi = highcut / nyq

        if hi == 1:
            wav_l = wav
        else:
            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            wav_l = sosfiltfilt(sos, wav)

            # downsample to the low sampling rate
            wav_l = resample_poly(wav_l, highcut * 2, self.hparams.audio.sampling_rate)
            # upsample to the original sampling rate
            wav_l = resample_poly(wav_l, self.hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = torch.zeros(fft_size, dtype = torch.int64)
        band[:int(hi * fft_size)] = 1

        return {
            'data_object': torch.from_numpy(wav).float(),
            'wav_l': torch.from_numpy(wav_l.copy()).float(),
            'band': band,
            'filename': self.data_list[index]
        }



def inf_loop(dataloader):
    """ Infinite dataloader loop for iteration-based training. """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    for transform_type, transforms in batch_transforms.items():
        if transforms is not None:
            for name in transforms:
                transforms[name] = transforms[name].to(device)


def get_dataloaders(config, device):
    train_ds = VCTKMultiSpkDataset(config, cv=0)
    val_ds   = VCTKMultiSpkDataset(config, cv=1)
    test_ds  = VCTKMultiSpkDataset(config, cv=2)
    print("Len train: ", len(train_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=set_worker_seed,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=set_worker_seed,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.dataloader.test_batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=set_worker_seed,
    )

    
    return {"train": train_dl, "test": test_dl}, None
