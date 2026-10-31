"""ninada (निनाद): SOTA Pan-Indic & Classical Sanskrit Neural Text-to-Speech."""
from .engine import SAMPLE_RATE, Ninada, encode_wav_bytes, speak, synthesize_klatt_pcm
from .g2p import (
    BLANK_ID,
    ID_TO_LANGUAGE,
    ID_TO_TOKEN,
    LANGUAGE_IDS,
    TOKEN_TO_ID,
    UNK_ID,
    VOCAB_TOKENS,
    WARMSTART_DONOR_MAP,
    apply_phonological_rules,
    brahmic_to_devanagari,
    detect_script_and_lang,
    iast_to_devanagari,
    segment_aksharas,
    text_to_ids,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "SAMPLE_RATE",
    "Ninada",
    "speak",
    "encode_wav_bytes",
    "synthesize_klatt_pcm",
    "VOCAB_TOKENS",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
    "BLANK_ID",
    "UNK_ID",
    "LANGUAGE_IDS",
    "ID_TO_LANGUAGE",
    "WARMSTART_DONOR_MAP",
    "detect_script_and_lang",
    "iast_to_devanagari",
    "brahmic_to_devanagari",
    "apply_phonological_rules",
    "segment_aksharas",
    "text_to_ids",
]
