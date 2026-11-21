"""dvitts command line.

  dvitts "नमस्ते, आप कैसे हैं?" --voice ananya -o hello.wav
  dvitts "Hello there." --prompt me.wav --prompt-text "exact words spoken in me.wav" -o out.wav
  dvitts --list-voices
  dvitts --detect clip.wav
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dvitts", description="DviTTS text-to-speech (en, hi, mr, bn, pa, gu, or)")
    ap.add_argument("text", nargs="?", help="text to speak")
    ap.add_argument("-o", "--out", default="dvitts.wav")
    ap.add_argument("--voice", default="ananya", help="preset voice id (see --list-voices)")
    ap.add_argument("--prompt", help="clone: 3-10 s wav of the voice")
    ap.add_argument("--prompt-text", help="clone: exact transcript of --prompt")
    ap.add_argument("--lang", help="en hi mr bn pa gu or (default: from the script; pass mr for Marathi)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="prakharkhatri/dvitts", help="HF repo id or local folder")
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    ap.add_argument("--no-watermark", action="store_true", help="research only: skip the AudioSeal mark")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--detect", metavar="WAV", help="print the probability that WAV carries the DviTTS watermark")
    a = ap.parse_args(argv)
    import soundfile as sf
    from .tts import DviTTS
    if a.detect:
        y, sr = sf.read(a.detect, dtype="float32")
        p = DviTTS.detect_watermark(y, sr)
        print(f"{p:.3f}", "marked" if p > 0.5 else "no mark found")
        return 0
    tts = DviTTS.from_pretrained(a.model, device=a.device, watermark=not a.no_watermark)
    if a.list_voices:
        for k, v in tts.voices().items():
            print(f"{k:12s} {v.get('lang', ''):3s} {v.get('gender', ''):7s} {v.get('source', '')}")
        return 0
    if not a.text:
        ap.error("text is required")
    if a.prompt:
        if not a.prompt_text:
            ap.error("--prompt needs --prompt-text (the exact words in the prompt)")
        w, sr = sf.read(a.prompt, dtype="float32", always_2d=True)
        v = tts.voice(w.mean(axis=1), sr, a.prompt_text)
    else:
        v = tts.preset(a.voice)
    y, sr = tts.say(v, a.text, seed=a.seed, lang=a.lang, watermark=not a.no_watermark)
    sf.write(a.out, y, sr)
    print(f"{a.out}: {len(y) / sr:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
