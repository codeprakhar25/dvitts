# DviTTS

Zero-shot text-to-speech for English and six Indian languages: Hindi, Marathi, Bengali, Punjabi, Gujarati and Odia.
Clone a voice from 3–10 s of speech, or use one of 18 named voices. Every output carries an AudioSeal watermark.

Stage 1 starts from [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) as its base
model (Apache 2.0), a 0.5B language model that already knows how to produce speech. We retrained it to emit
[IndiCodec](https://huggingface.co/prakharkhatri/IndiCodec) ids and to speak six Indian languages.

Weights, benchmarks and limits: the model card on [Hugging Face](https://huggingface.co/prakharkhatri/dvitts).
For phones and laptop CPUs see [DviTTS-Nano](https://github.com/codeprakhar25/dvitts-nano). Listen: [demo page](https://codeprakhar25.github.io/dvitts/).

## Install

```bash
python -m venv dvitts-env && source dvitts-env/bin/activate      # a fresh environment: the package pins its torch range
pip install "git+https://github.com/codeprakhar25/dvitts@v1.0.0"
```

Python 3.10+ and a CUDA GPU; installs PyTorch 2.6–2.8 (RTX 50-series needs 2.7+; not yet tested there). Tested on A10G: ~3.2 GB while speaking, 4 GB peak while cloning.

Windows: PyPI's Windows torch is CPU-only, so install the CUDA build first, then the package:
`pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`.

## Use

```bash
dvitts "नमस्ते, आप कैसे हैं?" --voice ananya -o hello.wav
dvitts "Please call me back after the meeting." --prompt me.wav --prompt-text "exact words spoken in me.wav" -o out.wav
dvitts --list-voices
dvitts --detect clip.wav          # is the DviTTS watermark in this file?
```

```python
import soundfile as sf
from dvitts import DviTTS

tts = DviTTS.from_pretrained("prakharkhatri/dvitts")
y, sr = tts.say("ananya", "कल सुबह हम स्टेशन पर मिलेंगे।")
sf.write("hello.wav", y, sr)

w, wsr = sf.read("me.wav", dtype="float32")
v = tts.voice(w, wsr, "exact words spoken in me.wav")
y, sr = tts.say(v, "The meeting starts at 10:30, so please be on time.")
```

`say()` normalises numbers, dates and money, splits long text at sentence ends, trims silence at the edges and adds
the watermark. `say_stream()` yields audio sentence by sentence. Devanagari text reads as Hindi; pass `lang="mr"` for
Marathi.

## Layout

| File | What |
|---|---|
| `dvitts/tts.py` | `DviTTS`: load, voices, `say`, `say_long`, `say_stream`, watermark |
| `dvitts/stage1.py` | 0.5B language model and its sampling loop (CUDA graph) |
| `dvitts/flow.py` | stage 2 flow model |
| `dvitts/frontend.py` | text normalisation and sentence split, 7 languages |
| `dvitts/codec/` | IndiCodec decoder/encoder (with the NeuCodec modules it needs) |

## Intended use and ethical considerations

For research and for building speech products with voices you have permission to use. Not for impersonation, fraud,
or cloning anyone's voice without their consent.

The preset voices are speakers from Rasa and IndicVoices-R (AI4Bharat) and LibriTTS-R, all CC BY 4.0, credited in
Acknowledgements. Display names (Ananya, Emma, …) are labels, not the speakers' names.

**Do not use these voices to imitate the real speakers, or to make anyone believe they are hearing them.** Clone only
voices you have permission to use. Audio from this model carries an AudioSeal watermark by default.

## Citation

If you use DviTTS, please cite:

```bibtex
@misc{khatri2026dvitts,
  title        = {DviTTS: Text-to-Speech for English and Six Indian Languages},
  author       = {Khatri, Prakhar},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/prakharkhatri/dvitts}}
}
```

DviTTS builds on CosyVoice 3 (its base model) and AudioSeal (its watermark); please cite them too:

```bibtex
@article{du2025cosyvoice,
  title   = {CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author  = {Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal = {arXiv preprint arXiv:2505.17589},
  year    = {2025}
}

@article{sanroman2024proactive,
  title   = {Proactive Detection of Voice Cloning with Localized Watermarking},
  author  = {San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D{\'e}fossez, Alexandre and Furon, Teddy and Tran, Tuan},
  journal = {ICML},
  year    = {2024}
}
```

## Acknowledgements

- **Base model:** [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (FunAudioLLM, Alibaba; Apache 2.0) and its Qwen2 text tokenizer.
- **Watermark:** [AudioSeal](https://github.com/facebookresearch/audioseal) (Meta; MIT). **Silence trim:** [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
- **Text front end:** [num2words](https://github.com/savoirfairelinux/num2words) (LGPL-2.1, used unmodified) and [indic-numtowords](https://github.com/AI4Bharat/indic-numtowords) (AI4Bharat; MIT); optional [NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing) (NVIDIA; Apache 2.0) for English.
- **Training data (CC BY 4.0):** Emilia-YODAS English subset (Amphion), LibriTTS and LibriTTS-R, VCTK, and from AI4Bharat Rasa, IndicVoices-R and IndicVoices (training split).
- **Preset voices:** speakers from Rasa, IndicVoices-R and LibriTTS-R (CC BY 4.0). They belong to those speakers and must not be used to impersonate them.

## Licence

Apache 2.0 (LICENSE). The bundled NeuCodec modules keep their own LICENSE in `dvitts/codec/neucodec/`.
