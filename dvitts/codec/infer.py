"""IndiCodec inference. Self-contained; the release file needs no weight download.

    from indicodec import IndiCodec
    c = IndiCodec.load("model.safetensors")
    idx_s, idx_a = c.encode(wav, sr)
    wav24 = c.decode(idx_s, idx_a, n_samples=n24)
    wav24 = c.reconstruct(wav, sr)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SPEC = json.loads((_HERE / "codec_spec.json").read_text())


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def n24_of(wav, sr: int) -> int:
    return int(round(len(wav) * _SPEC["sample_rate_out"] / sr))


def _refuse_train_flags(st: dict) -> None:
    bad = []
    if st.get("acoustic_continuous"):
        bad.append("acoustic_continuous")
    if int(st.get("acoustic_frames") or 1) != 1:
        bad.append(f"acoustic_frames={st.get('acoustic_frames')}")
    if st.get("rope_fix"):
        bad.append("rope_fix")
    if int(st.get("phase2d") or 0):
        bad.append(f"phase2d={st.get('phase2d')}")
    if bad:
        raise SystemExit(f"IndiCodec infer refuses {bad}: not an IndiCodec release checkpoint")


def read_ckpt(p: Path) -> tuple[dict, str]:
    """md5-check a weight file against the spec and return ({"gen", "tail_frames", "levels_a", ...}, md5)."""
    got = md5_file(p)
    if p.suffix != ".safetensors":
        raise SystemExit(f"{p}: IndiCodec loads the release model.safetensors only")
    want = _SPEC["md5_safetensors"]
    if got != want:
        raise SystemExit(f"{p} md5 {got} != spec {want}")
    from safetensors import safe_open
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        gen = {k: f.get_tensor(k) for k in f.keys()}
    st = {"gen": gen, "meta": meta, "tail_frames": int(meta.get("tail_frames", 0)),
          "levels_a": [int(x) for x in meta.get("levels_a", "").split(",") if x]}
    return st, got


class IndiCodec:
    def __init__(self, model, device: str, amp: bool, ckpt_md5: str):
        self.model = model
        self.device = device
        self.amp = amp
        self.ckpt_md5 = ckpt_md5
        self.spec = _SPEC

    @classmethod
    def load(cls, ckpt: str | Path, device: str | None = None, amp: bool | None = None) -> "IndiCodec":
        try:
            from .model import IndiNet
        except ImportError:
            from model import IndiNet  # folder-alone: infer.py + model.py side by side

        p = Path(ckpt)
        st, got = read_ckpt(p)
        if int(st.get("tail_frames", 0)) != int(_SPEC["tail_frames"]):
            raise SystemExit(f"ckpt tail_frames {st.get('tail_frames')} != spec {_SPEC['tail_frames']}")
        if list(st.get("levels_a") or []) != list(_SPEC["acoustic"]["levels"]):
            raise SystemExit(f"ckpt levels_a {st.get('levels_a')} != spec {_SPEC['acoustic']['levels']}")
        _refuse_train_flags(st)
        bundled = any(k.startswith("semantic_model.") for k in st["gen"])   # model.safetensors bundles the w2v-BERT layers
        m = IndiNet(levels_a=st.get("levels_a"), w2v_bundled=bundled)
        m.load_gen(st["gen"])
        m.tail_frames = int(st.get("tail_frames", 0))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(device).eval()
        if amp is None:
            amp = device != "cpu"
        return cls(m, device, amp, got)

    def encode(self, wav, sr: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            from .model import encode_wav, _mod_dev
        except ImportError:
            from model import encode_wav, _mod_dev
        enc = getattr(self.model, "semantic_model", None)
        enc_dev = str(_mod_dev(enc)) if enc is not None else self.device
        if enc_dev.startswith("cuda"):
            enc_dev = "cuda"
        idx_s, idx_a, _ = encode_wav(self.model, wav, sr, enc_dev, amp=self.amp)
        return idx_s.detach().cpu().numpy(), idx_a.detach().cpu().numpy()

    @torch.no_grad()
    def decode(self, idx_s, idx_a, n_samples: int | None = None) -> np.ndarray:
        try:
            from .model import _mod_dev
        except ImportError:
            from model import _mod_dev
        bb = self.model.backbone
        use_amp = self.amp and str(_mod_dev(bb)).startswith("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            y = self.model.decode_codes(torch.as_tensor(idx_s), torch.as_tensor(idx_a), mode="full").float().cpu().numpy().reshape(-1)
        if n_samples is None:
            return y
        return y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))

    def reconstruct(self, wav, sr: int) -> np.ndarray:
        return self.decode(*self.encode(wav, sr), n_samples=n24_of(wav, sr))

    def deterministic(self, on: bool = True) -> None:
        if on:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = on
        torch.backends.cudnn.benchmark = not on
        torch.use_deterministic_algorithms(on, warn_only=True)
