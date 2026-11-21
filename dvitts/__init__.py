"""DviTTS: zero-shot text-to-speech for English and six Indian languages."""
from .tts import SR, DviTTS, Voice, check_text, unsupported_scripts

__version__ = "1.0.0"
__all__ = ["DviTTS", "Voice", "SR", "check_text", "unsupported_scripts", "__version__"]
