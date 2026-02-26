import torch
from typing import List, Dict, Any

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    wavs   = [sample["data_object"] for sample in batch]
    wav_ls = [sample["wav_l"]       for sample in batch]
    bands  = [sample["band"]        for sample in batch]

    # pad wavs & wav_ls to the same length
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
    }
