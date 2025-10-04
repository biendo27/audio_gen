import os
from typing import Iterable, List, Optional

import torch
import torchaudio


def concatenate_wavs(segment_paths: Iterable[str], output_path: str) -> None:
    """Concatenate wav files in order and write to output_path."""
    segment_paths = [str(path) for path in segment_paths]
    if not segment_paths:
        raise ValueError("No segment paths provided for concatenation.")

    waveforms: List[torch.Tensor] = []
    sample_rate: Optional[int] = None

    for path in segment_paths:
        waveform, sr = torchaudio.load(path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch for {path}: expected {sample_rate}, got {sr}")
        waveforms.append(waveform)

    combined = torch.cat(waveforms, dim=1)
    torchaudio.save(output_path, combined, sample_rate or 22050)


def safe_remove_files(paths: Iterable[str], log=None) -> None:
    """Remove files if they exist, suppressing errors."""
    for path in paths:
        try:
            os.remove(path)
            if log:
                log(f"[INFO] Removed temp file {path}")
        except FileNotFoundError:
            continue
        except Exception as err:
            if log:
                log(f"[WARN] Could not remove {path}: {err}")
