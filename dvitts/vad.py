"""Silence at the edges and inside a clip, from Silero VAD (an energy floor cannot tell a steady drone from speech)."""
from __future__ import annotations

import numpy as np
import torch


def dead_air(y: np.ndarray, sr: int, model, get_ts) -> dict:
    """-> {"dur", "lead", "tail", "gap", "speech"} in seconds; gap = longest non-speech stretch inside the clip."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import torchaudio.functional as AF
        y = AF.resample(torch.from_numpy(y), sr, 16000).numpy()
    ts = get_ts(torch.from_numpy(y), model, sampling_rate=16000, return_seconds=True, min_silence_duration_ms=300, speech_pad_ms=30)
    dur = len(y) / 16000
    if not ts:
        return {"dur": round(dur, 2), "lead": round(dur, 2), "tail": 0.0, "gap": 0.0, "speech": 0.0}
    gaps = [b["start"] - a["end"] for a, b in zip(ts, ts[1:])]
    return {"dur": round(dur, 2), "lead": round(ts[0]["start"], 2), "tail": round(dur - ts[-1]["end"], 2),
            "gap": round(max(gaps, default=0.0), 2), "speech": round(sum(t["end"] - t["start"] for t in ts), 2)}
