"""DviTTS: zero-shot text-to-speech for English and six Indian languages (Hindi, Marathi, Bengali, Punjabi, Gujarati, Odia).

    from dvitts import DviTTS
    tts = DviTTS.from_pretrained("prakharkhatri/dvitts")          # CUDA GPU; codec from prakharkhatri/IndiCodec
    v = tts.preset("ananya")                                       # or tts.voice(wav, sr, transcript)
    y, sr = tts.say(v, "नमस्ते, आप कैसे हैं?")

One call: text front end (numbers, dates, money spelled out) -> stage 1 (0.5B LM, content ids at 25 Hz) -> stage 2 (flow,
acoustic ids at 50 Hz) -> IndiCodec decoder (24 kHz) -> edge trim -> AudioSeal watermark. Long text is split into
sentence chunks and joined with 0.2 s of silence.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import frontend as fe
from . import flow as fl
from .stage1 import INSTRUCT, ENDOFPROMPT_ID, FastS1, load_s1
from .vad import dead_air

SR = 24_000
REF_MAX = 500                     # prompt frames at 50 Hz (10 s)
CODEC_REPO, CODEC_MD5 = "prakharkhatri/IndiCodec", "7d9f2d7d"
TEMP, CAP_X, NFE, CFG, S2_NOISE, TRIM_PAD = 0.8, 3.0, 16, 4.0, 0.7, 0.1
REDRAWS, REDRAW_SEED = 1, 1_000_003      # stage 1 drawn again (new seed) when it reaches the length cap without ending
GAP_RETRY_S, GAP_SEED = 2.0, 7_000_001   # a take with a non-speech stretch >= 2 s inside is made again; shorter gap kept
MAX_S = 30.0                      # ceiling per call (stage 1 trained on clips of up to 30 s)
PROMPT_MIN_S, PROMPT_MAX_S = 3.0, 10.0
SPLIT_S, RATE_SCALE = 25.0, 0.8   # long text: chunks of at most ~25 s of expected speech
MIN_RATIO = 2.0                   # no end of speech before 2 x target text tokens, scaled per script below
MIN_SCALE = {"latin": 1.0, "deva": 1.0, "beng": 1.0, "guru": 10.2 / 18.7, "gujr": 10.2 / 19.2, "orya": 10.2 / 24.5}
TOK_S = {"latin": 2.8, "deva": 9.0, "beng": 10.0, "guru": 14.0, "gujr": 15.0, "orya": 18.0}   # text tokens / s of speech
WM_MSG = 0xB1C5                   # AudioSeal 16-bit message
WM_CPU_BUCKET = 8000
FIRST_S, GAP_S = 3.0, 0.2
_SCRIPT = [("deva", 0x0900, 0x097F), ("beng", 0x0980, 0x09FF), ("guru", 0x0A00, 0x0A7F), ("gujr", 0x0A80, 0x0AFF),
           ("orya", 0x0B00, 0x0B7F)]
_LANG_OF = {"latin": "en", "deva": "hi", "beng": "bn", "guru": "pa", "gujr": "gu", "orya": "or"}
SPOKEN = "English, Hindi, Marathi, Bengali, Punjabi, Gujarati, Odia"


def _runs(text: str) -> list[tuple[str, str]]:
    """Text -> [(script, run)]: Latin letters vs each Indic block; digits/punctuation/space join the current run."""
    out: list[tuple[str, str]] = []
    for ch in text:
        o = ord(ch)
        sc_ = next((n for n, a, b in _SCRIPT if a <= o <= b), "latin" if ch.isalpha() else None)
        if out and (sc_ is None or sc_ == out[-1][0]):
            out[-1] = (out[-1][0], out[-1][1] + ch)
        elif sc_ is not None:
            out.append((sc_, ch))
        elif not out:
            out.append(("latin", ch))
    return out


def unsupported_scripts(text: str) -> list[str]:
    """Letters from scripts the model never trained on (e.g. Tamil, Telugu, Kannada, Malayalam)."""
    bad = set()
    for ch in text:
        o = ord(ch)
        if not ch.isalpha() or o <= 0x024F or any(a <= o <= b for _, a, b in _SCRIPT) or 0x1E00 <= o <= 0x1EFF:
            continue
        bad.add(unicodedata.name(ch, f"U+{o:04X}").split()[0])
    return sorted(bad)


def check_text(text: str) -> None:
    bad = unsupported_scripts(text)
    if bad:
        raise ValueError(f"Unsupported script: {', '.join(bad)}. This model speaks {SPOKEN}.")


def min_ratio(run_tokens: list[tuple[str, int]], n_text: int) -> float:
    if all(MIN_SCALE.get(sc_, 1.0) >= 1.0 for sc_, _ in run_tokens) or n_text <= 0:
        return MIN_RATIO
    eff = sum(n * min(1.0, MIN_SCALE.get(sc_, 1.0)) for sc_, n in run_tokens)
    return MIN_RATIO * min(1.0, eff / n_text)


def guess_lang(text: str) -> str:
    """Front-end language from the script with the most letters (Devanagari -> hi; pass lang="mr" for Marathi)."""
    n: dict[str, int] = {}
    for sc_, run in _runs(text):
        n[sc_] = n.get(sc_, 0) + sum(ch.isalpha() or 0x0900 <= ord(ch) <= 0x0B7F for ch in run)
    return _LANG_OF[max(n, key=n.get)] if n else "en"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _to16(wav, sr) -> np.ndarray:
    import torchaudio.functional as AF
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    return (AF.resample(x[None], sr, 16_000)[0] if sr != 16_000 else x).numpy()


def _frames50(s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.repeat(s.reshape(-1), 2)
    a = a.reshape(-1)
    n = min(len(a), len(c))
    return c[:n], a[:n]


@dataclass
class Voice:
    ref_text_ids: list[int]
    ref_ids: list[int]          # stage-1 prompt: content ids at 25 Hz (first 10 s)
    n_ref_ids: int
    c_r: np.ndarray             # stage-2 prompt frames (50 Hz)
    ac_r: np.ndarray
    ms: dict = field(default_factory=dict)
    warn: str = ""              # set by voice() for a short or cut prompt


class DviTTS:
    """Load with DviTTS.from_pretrained(repo_id_or_dir). device "cuda" (default; tested on A10G) or "cpu" (slow)."""

    def __init__(self, pack: str | Path, device: str | None = None, trim: bool = True, watermark: bool = True):
        self.pack = _resolve(pack)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not _have_cc():
            # torch.compile (used inside AudioSeal and its dependencies) needs a C compiler to build GPU kernels; without
            # one, run everything eagerly. Same model, same math; only the kernels are not fused.
            torch._dynamo.config.disable = True
        self.cfg = json.loads((self.pack / "config.json").read_text(encoding="utf-8"))
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(self.pack / "tokenizer.json"))
        if self.encode("<|endofprompt|>") != [ENDOFPROMPT_ID]:
            raise RuntimeError("tokenizer.json does not match the CosyVoice3 text tokenizer")
        self.instr = self.encode(INSTRUCT)
        self.s1 = self._load_s1()
        dt = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.fast = FastS1(self.s1, self.dev, dtype=dt, mode="graph" if self.dev == "cuda" else "trace")
        self.codec = self._load_codec()
        self.s2, self.basis = self._load_f1()
        self.trim = trim
        self._vad = None
        if trim:
            from silero_vad import get_speech_timestamps, load_silero_vad
            self._vad = (load_silero_vad(), get_speech_timestamps)
        self.wm = None
        if watermark:
            from audioseal import AudioSeal
            self.wm = AudioSeal.load_generator("audioseal_wm_16bits").eval().to(self.dev)
        self._index = json.loads((self.pack / "voices" / "index.json").read_text(encoding="utf-8"))

    @classmethod
    def from_pretrained(cls, repo_or_dir: str | Path = "prakharkhatri/dvitts", **kw) -> "DviTTS":
        return cls(repo_or_dir, **kw)

    # ---- loading ----------------------------------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def _load_s1(self):
        from safetensors.torch import load_file
        cfg = json.loads((self.pack / "s1_config.json").read_text())
        return load_s1(load_file(str(self.pack / "s1.safetensors"), device="cpu"), cfg).to(self.dev)

    def _load_f1(self):
        from safetensors.torch import load_file
        sd = load_file(str(self.pack / "f1.safetensors"), device="cpu")
        cfg = json.loads((self.pack / "f1_config.json").read_text())
        m = fl.FlowStage2(torch.zeros(6561, 1024), d=int(cfg["d"]), layers=int(cfg["layers"]), heads=int(cfg["heads"]),
                          qk_norm=bool(cfg["qk_norm"]), meanflow=bool(cfg.get("meanflow", False)))
        m.load_state_dict(sd)
        basis = torch.from_numpy(np.load(self.pack / "f1_basis.npy")).long().to(self.dev)
        return m.to(self.dev).eval(), basis

    def _load_codec(self):
        """IndiCodec v1 from its own repo. On GPU the decoder (Vocos) runs in bf16 on the GPU; the encoder (w2v-BERT,
        only used to clone a new voice) stays in CPU RAM and visits the GPU for each voice() call."""
        from huggingface_hub import hf_hub_download
        from .codec import IndiCodec
        p = Path(hf_hub_download(CODEC_REPO, "model.safetensors"))
        if _md5(p)[:8] != CODEC_MD5:
            raise RuntimeError(f"IndiCodec {p} md5 {_md5(p)[:8]}, expected {CODEC_MD5}")
        codec = IndiCodec.load(p, device="cpu")
        if self.dev == "cuda":
            torch.backends.cudnn.benchmark = False
            m = codec.model
            for name in ("backbone", "head"):
                setattr(m, name, getattr(m, name).to(device="cuda", dtype=torch.bfloat16))
            codec.device = "cuda"
            torch.cuda.empty_cache()
        return codec

    _ENC = ("semantic_model", "CodecEnc", "SemanticEncoder_module", "fsq_s", "fsq_a")

    def _encoder_to(self, device: str) -> None:
        m = self.codec.model
        for name in self._ENC:
            mod = getattr(m, name, None)
            if mod is not None:
                setattr(m, name, mod.to(device))
        if str(device).startswith("cpu") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encoder_on_cpu(self) -> bool:
        sm = getattr(self.codec.model, "semantic_model", None)
        for x in (*sm.parameters(), *sm.buffers()) if sm is not None else ():
            return x.device.type == "cpu"
        return False

    # ---- voices -----------------------------------------------------------------------------------------------------
    def voices(self) -> dict:
        """Named preset voices: id -> {label, lang, gender, source, transcript}."""
        return self._index

    def preset(self, name: str) -> Voice:
        if name not in self._index:
            raise KeyError(f"no preset {name!r}; one of {', '.join(self._index)}")
        z = np.load(self.pack / "voices" / f"{name}.npz")
        return Voice(z["ref_text_ids"].astype(np.int64).tolist(), z["ref_ids"].astype(np.int64).tolist(),
                     int(z["n_ref_ids"]), np.asarray(z["c_r"]), np.asarray(z["ac_r"]))

    def fit_prompt(self, wav: np.ndarray, sr: int, max_s: float = PROMPT_MAX_S) -> np.ndarray:
        """A prompt longer than 10 s is cut at the last pause before 10 s (the model sees at most 10 s of prompt)."""
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) <= max_s * sr:
            return wav
        cut = int(max_s * sr)
        if self._vad is not None:
            vad, get_ts = self._vad
            ts = get_ts(torch.from_numpy(_to16(wav, sr)), vad, sampling_rate=16000, min_silence_duration_ms=150)
            ends = [t["end"] / 16000 for t in ts if t["end"] / 16000 <= max_s - .1]
            if ends and ends[-1] >= 3.0:
                cut = int((ends[-1] + .1) * sr)
        return wav[:cut]

    @torch.no_grad()
    def voice(self, wav: np.ndarray, sr: int, ref_text: str, transcribe=None) -> Voice:
        """Clone: 3-10 s of clean speech (mono float array) and its exact transcript -> Voice. A longer prompt is cut at a
        pause; pass transcribe(wav, sr) -> text to re-transcribe the kept part (else the transcript no longer matches)."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        cut_warn = ""
        if len(wav) > PROMPT_MAX_S * sr:
            n0 = len(wav) / sr
            wav = self.fit_prompt(wav, sr)
            if transcribe is not None:
                ref_text = transcribe(wav, sr)
            else:
                cut_warn = (f"Prompt cut from {n0:.1f} to {len(wav) / sr:.1f} s (the model uses at most {PROMPT_MAX_S:.0f} s). "
                            "The transcript given covers the full clip: pass transcribe=, or call fit_prompt() and transcribe "
                            "its output, else the voice may say extra words first.")
                warnings.warn(cut_warn, stacklevel=2)
        t0 = time.perf_counter()
        off = self.dev == "cuda" and self._encoder_on_cpu()
        if off:
            self._encoder_to("cuda")
        try:
            s_r, a_r = self.codec.encode(wav, sr)
        finally:
            if off:
                self._encoder_to("cpu")
        flat = np.asarray(s_r).reshape(-1)
        c_r, ac_r = _frames50(s_r, a_r)
        v = Voice(self.encode(" " + ref_text), flat[: REF_MAX // 2].astype(np.int64).tolist(), len(flat),
                  c_r[:REF_MAX], ac_r[:REF_MAX])
        v.ms["encode"] = (time.perf_counter() - t0) * 1e3
        v.warn = cut_warn
        if len(wav) / sr < PROMPT_MIN_S:
            v.warn = (v.warn + " " if v.warn else "") + (
                f"Prompt is {len(wav) / sr:.1f} s. Under {PROMPT_MIN_S:.0f} s the voice match and the words get worse. "
                "Use 5-10 s of clean speech.")
            warnings.warn(v.warn, stacklevel=2)
        return v

    # ---- speech -----------------------------------------------------------------------------------------------------
    def _exp_ids(self, text: str) -> float:
        return 25 * sum(len(self.encode(" " + r)) / TOK_S.get(n, 2.8) for n, r in _runs(text))

    def _gen(self, seed: int) -> torch.Generator:
        g = torch.Generator(device="cuda") if self.dev == "cuda" else torch.Generator()
        return g.manual_seed(seed)

    @torch.no_grad()
    def say(self, v: Voice | str, text: str, seed: int = 0, lang: str | None = None, normalize: bool = True,
            watermark: bool = True, timings: dict | None = None) -> tuple[np.ndarray, int]:
        """Text -> (24 kHz mono float32 audio, 24000). v: a Voice or a preset name. lang: "en", "hi", "mr", "bn", "pa",
        "gu", "or" (default from the script; Devanagari reads as Hindi, so pass "mr" for Marathi). Text longer than one
        chunk is split at sentence ends (say_long)."""
        v = self.preset(v) if isinstance(v, str) else v
        check_text(text)
        lang = lang or guess_lang(text)
        if normalize:
            text = fe.normalize(text, lang)
        tm = {} if timings is None else timings
        if len(text) > fe.MAX_CHARS.get(lang, fe.MAX_CHARS_IN) or RATE_SCALE * self._exp_ids(text) / 25 > SPLIT_S:
            return self.say_long(v, text, lang, seed=seed, normalize=False, timings=tm, watermark=watermark)
        y, sr = self._say_one(v, text, seed=seed, timings=tm)
        return (self._mark(y, tm) if watermark else y), sr

    @torch.no_grad()
    def _say_one(self, v: Voice, text: str, seed: int = 0, timings: dict | None = None,
                 gap_retry: bool = True) -> tuple[np.ndarray, int]:
        tm = {} if timings is None else timings
        tt = self.encode(" " + text)
        exp_ids = self._exp_ids(text)
        cap = min(int(CAP_X * exp_ids), int(MAX_S * 25)) if CAP_X > 0 else int(MAX_S * 25)
        mr = min_ratio([(n, len(self.encode(" " + r))) for n, r in _runs(text)], len(tt))
        prefix = self.instr + v.ref_text_ids + tt
        t0 = time.perf_counter()
        ids, eos = self._s1_draw(prefix, v, len(tt), self._gen(seed), mr, cap)
        tm["redraw"] = 0
        if not eos and REDRAWS > 0:
            for r in range(1, REDRAWS + 1):
                ids2, eos2 = self._s1_draw(prefix, v, len(tt), self._gen(seed + r * REDRAW_SEED), mr, cap)
                tm["redraw"] = r
                if eos2:
                    ids, eos = ids2, eos2
                    break
        tm["s1_ms"], tm["n_ids"], tm["eos"] = (time.perf_counter() - t0) * 1e3, len(ids), eos
        n24 = len(ids) * (SR // 25)
        if len(ids) < 5:
            return np.zeros(n24, dtype=np.float32), SR
        c_t = np.repeat(np.asarray(ids, dtype=np.int64), 2)
        t0 = time.perf_counter()
        a_hat = self._s2_ids(v.c_r, v.ac_r, c_t, seed)
        tm["f1_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        y = self._decode(np.asarray(ids), a_hat, n24)
        tm["dec_ms"] = (time.perf_counter() - t0) * 1e3
        if self.trim:
            r = dead_air(y, SR, *self._vad)
            y = np.asarray(y, dtype=np.float32)
            y = y[max(0, int((r["lead"] - TRIM_PAD) * SR)):min(len(y), int((r["dur"] - r["tail"] + TRIM_PAD) * SR))]
            tm["gap"], tm["gap_retry"] = r["gap"], 0
            if gap_retry and GAP_RETRY_S > 0 and r["gap"] >= GAP_RETRY_S:
                t2: dict = {}
                y2, _ = self._say_one(v, text, seed=seed + GAP_SEED, timings=t2, gap_retry=False)
                if t2["gap"] < r["gap"]:
                    tm.update(t2)
                    y = y2
                tm["gap_retry"] = 1
        tm["audio_s"] = len(y) / SR
        return np.asarray(y, dtype=np.float32), SR

    def _s1_draw(self, prefix, v: Voice, n_text: int, gen, mr: float, cap: int):
        ctx = torch.autocast("cuda", enabled=False) if self.dev == "cuda" else contextlib.nullcontext()
        with ctx:
            return self.fast.generate(prefix, v.ref_ids, n_text, gen, min_ratio=mr, max_ids=cap, temp=TEMP)

    def _s2_ids(self, c_r, ac_r, c_t, seed: int) -> np.ndarray:
        t = lambda x: torch.as_tensor(np.asarray(x).astype(np.int64), device=self.dev)  # noqa: E731
        dig_ref = fl.ids_to_digits(t(ac_r), self.basis)
        dig = fl.generate_flow(self.s2, t(c_r), dig_ref, t(c_t), nfe=NFE, w=CFG, gen=self._gen(seed), noise=S2_NOISE)
        return fl.digits_to_ids(dig, self.basis).cpu().numpy()

    def _decode(self, s, a, n: int) -> np.ndarray:
        m = self.codec.model
        dev = getattr(self.codec, "device", "cuda")
        st = torch.as_tensor(np.asarray(s).reshape(1, -1).astype(np.int64), device=dev)
        at = torch.as_tensor(np.asarray(a).reshape(1, -1).astype(np.int64), device=dev)
        amp = bool(getattr(self.codec, "amp", True)) and str(dev).startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = m.decode_codes(st, at, mode="full").float().cpu().numpy().reshape(-1)
        return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))

    def _mark(self, y: np.ndarray, tm: dict) -> np.ndarray:
        """AudioSeal 16 kHz residual, resampled to 24 kHz and added."""
        if self.wm is None or len(y) == 0:
            return y
        import torchaudio.functional as AF
        t0 = time.perf_counter()
        x16 = AF.resample(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None], SR, 16000)
        ctx = _wm_ctx(self.dev)
        if self.dev == "cpu":
            x16 = torch.nn.functional.pad(x16, (0, -x16.shape[-1] % WM_CPU_BUCKET))
        msg = torch.tensor([[int(b) for b in format(WM_MSG, "016b")]], device=self.dev)
        with torch.no_grad(), ctx:
            r16 = self.wm.get_watermark(x16.to(self.dev)[None], 16000, message=msg)[0].cpu()
        r = AF.resample(r16, 16000, SR)[0].numpy()[: len(y)]
        tm["wm_ms"] = (time.perf_counter() - t0) * 1e3
        return (y + np.pad(r, (0, len(y) - len(r)))).astype(np.float32)

    @staticmethod
    def detect_watermark(y: np.ndarray, sr: int) -> float:
        """Probability that y carries the AudioSeal mark."""
        from audioseal import AudioSeal
        import torchaudio.functional as AF
        y = np.asarray(y, dtype=np.float32)
        y = y.mean(axis=1) if y.ndim > 1 else y
        det = AudioSeal.load_detector("audioseal_detector_16bits").eval()
        x16 = AF.resample(torch.from_numpy(np.ascontiguousarray(y))[None], sr, 16000)
        with torch.no_grad(), _wm_ctx("cpu"):
            p, _ = det.detect_watermark(x16[None], 16000)
        return float(p)

    def _chunks(self, text: str, lang: str, normalize: bool) -> list[str]:
        t = fe.normalize(text, lang) if normalize else text
        out = []
        for c in fe.split(t, lang):
            e = RATE_SCALE * self._exp_ids(c) / 25
            if e <= SPLIT_S:
                out.append(c)
                continue
            mx = max(40, int(len(c) / -(-e // SPLIT_S)))
            out += fe.split(c, lang, max_chars=mx, min_chars=int(mx * .6))
        return out

    @torch.no_grad()
    def say_long(self, v: Voice | str, text: str, lang: str | None = None, seed: int = 0, normalize: bool = True,
                 gap_s: float = GAP_S, timings: dict | None = None, watermark: bool = True) -> tuple[np.ndarray, int]:
        """Any length: sentence chunks (chunk i uses seed * 1000 + i), each edge-trimmed, joined with gap_s of silence."""
        v = self.preset(v) if isinstance(v, str) else v
        check_text(text)
        lang = lang or guess_lang(text)
        tm = {} if timings is None else timings
        tm["chunks"], parts = [], []
        chunks = self._chunks(text, lang, normalize)
        gap = np.zeros(int(gap_s * SR), dtype=np.float32)
        for i, c in enumerate(chunks):
            ct: dict = {}
            y, _ = self._say_one(v, c, seed=seed * 1000 + i, timings=ct)
            tm["chunks"].append({"text": c, **ct})
            parts += [y, gap] if i < len(chunks) - 1 else [y]
        y = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if watermark:
            y = self._mark(y, tm)
        return y, SR

    def say_stream(self, v: Voice | str, text: str, lang: str | None = None, seed: int = 0, normalize: bool = True,
                   gap_s: float = GAP_S, first_s: float = FIRST_S, watermark: bool = True):
        """Sentence-level streaming: yields (audio, sr, info) per chunk as soon as it is made. The first chunk is kept
        short (<= first_s of expected speech) so playback starts early. Concatenated chunks equal say_long."""
        v = self.preset(v) if isinstance(v, str) else v
        check_text(text)
        lang = lang or guess_lang(text)
        t = fe.normalize(text, lang) if normalize else text
        chunks = fe.split(t, lang)
        if chunks:
            e = RATE_SCALE * self._exp_ids(chunks[0]) / 25
            if e > first_s:
                mx = max(30, int(len(chunks[0]) * first_s / e))
                chunks = fe.split(chunks[0], lang, max_chars=mx, min_chars=int(mx * .6)) + chunks[1:]
        gap = np.zeros(int(gap_s * SR), dtype=np.float32)
        t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            ct: dict = {}
            y, _ = self._say_one(v, c, seed=seed * 1000 + i, timings=ct)
            if watermark:
                y = self._mark(y, ct)
            if i < len(chunks) - 1:
                y = np.concatenate([y, gap])
            yield y, SR, {"i": i, "n": len(chunks), "text": c, "t_ready_s": time.perf_counter() - t0}


def _have_cc() -> bool:
    import os
    import shutil
    return bool(os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc"))


def _wm_ctx(dev: str):
    """AudioSeal compiles its kernels with torch.compile, which needs a C compiler. Keep it on GPU when one is installed
    (the tested path); otherwise, and always on CPU, run it uncompiled."""
    import os
    import shutil
    ctx = contextlib.ExitStack()
    if dev == "cpu" or not _have_cc():
        from audioseal.libs.moshi.utils.compile import no_compile
        ctx.enter_context(no_compile())
    return ctx


def _resolve(pack: str | Path) -> Path:
    p = Path(pack)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    # model.safetensors is a copy of s1.safetensors kept for the Hub's parameter count; never download it twice
    return Path(snapshot_download(str(pack), ignore_patterns=["model.safetensors", "*.md", ".gitattributes"]))
