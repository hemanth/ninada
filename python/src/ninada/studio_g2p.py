"""
Ninada v4.0 — Native Per-Language 24,000 Hz Studio VITS Character Tokenizer.

Maps native Unicode orthography directly to each specialist's exact character vocabulary
(`sa`: 144, `hi`: 120, `kn`: 85, `ta`: 61, `te`: 110, `ml`: 101 tokens) with zero Hindi pivot,
zero lossy transliteration when input is in native script, and automatic IAST / cross-script
fallback when Roman or cross-script text is supplied.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .g2p import detect_script_and_lang, iast_to_devanagari, to_native_model_text

STUDIO_HOP_LENGTH = 256
STUDIO_SAMPLE_RATE = 24000

STUDIO_SPECS: dict[str, dict[str, Any]] = {
    "sa": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": ')ऊभॐब८•औॲ?z५ी/ऩठअवइ\',डळत3° ञोnआफ़ॊच:․छऍई;ओuगझल0२ॆऑऱ!ॕ"(मर|9…न-ॄु5ृहेॅःएघ७शऎ1ङॠं।१ष४cॉधाऽ\u200aकि2.ऋउ–ूa्स६जखदथ०68णऐैफ़ँs९३4ौ7—टपढयऌॡॢॣ॥',
        "blank": "<BLNK>",
    },
    "hi": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": '८ढी५ू)बख”ंनग़ँक़ङत1।व2ट(ऑओग़फज9्थुख़०छञड5ड़औ१ैचफ़रण?ॐ२लॠदय़म‘–"6भॉ7’ईय४७इशोठ3९उ“ज़ष३िौ4ऊःहढ़ऋअॅआऐए0क६घझधस,\'ृपे 8ा',
        "blank": "<BLNK>",
    },
    "kn": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": 'ಆಽ*ಬಚಠಸಂುಣೂೆಖಔಢಭ(ಡಿಝದಈಫೊ ಛಅಐಗಉ?ನೃಧಇೀಜಥೌತಊಾೇೕಹಷಕಲೈಒಳಯ,ಟಪಓಎ\'೦ಋ"್ಶರಘೋಞಏವಮ',
        "blank": "<BLNK>",
    },
    "ta": {
        "pad": "_",
        "punctuations": ".,!?; ",
        "characters": "ஂஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்ௗ'-()",
        "blank": "<BLNK>",
    },
    "te": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": 'ఆధఏఘbః…ఈడమఖకలఊ’దఒ,:తీఞ·గైం.ొిఔ•్ృయ।ఱూ/[ఢేసఝ;\'ఛు-టనళఓఐథ“భపాషఋఙోఠ–ఉణ!జబ("రa”ఎఁశఇఫ|అహ*—ౌ?ʼ)఼‘ వచె౦',
        "blank": "<BLNK>",
    },
    "ml": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": "ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഽാിീുൂൃെേൈൊോൌ്ൗൠൡ൦൧൨൩൪൫൬൭൮൯ൺൻർൽൾൿ",
        "blank": "<BLNK>",
    },
}


def _build_studio_vocabs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lc, spec in STUDIO_SPECS.items():
        vocab = [spec["pad"]] + list(spec["punctuations"]) + list(spec["characters"]) + [spec["blank"]]
        c2i = {ch: idx for idx, ch in enumerate(vocab)}
        out[lc] = {
            "lang": lc,
            "vocab": vocab,
            "char_to_id": c2i,
            "pad_id": c2i[spec["pad"]],
            "blank_id": c2i[spec["blank"]],
            "num_chars": len(vocab),
        }
    return out


STUDIO_VOCABS: dict[str, dict[str, Any]] = _build_studio_vocabs()

_WHITESPACE_RE = re.compile(r"\s+")
_AUX_SYMBOLS_RE = re.compile(r"[\<\>\(\)\[\]\"]+")
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "sa": (0x0900, 0x097F),
    "hi": (0x0900, 0x097F),
    "kn": (0x0C80, 0x0CFF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "ml": (0x0D00, 0x0D7F),
}
_MALAYALAM_CHILLU_SEQ = {
    "\u0d23\u0d4d\u200d": "\u0d7a",
    "\u0d28\u0d4d\u200d": "\u0d7b",
    "\u0d30\u0d4d\u200d": "\u0d7c",
    "\u0d32\u0d4d\u200d": "\u0d7d",
    "\u0d33\u0d4d\u200d": "\u0d7e",
    "\u0d15\u0d4d\u200d": "\u0d7f",
    "\u0d23\u0d4d\u200c": "\u0d7a",
    "\u0d28\u0d4d\u200c": "\u0d7b",
    "\u0d30\u0d4d\u200c": "\u0d7c",
    "\u0d32\u0d4d\u200c": "\u0d7d",
    "\u0d33\u0d4d\u200c": "\u0d7e",
    "\u0d15\u0d4d\u200c": "\u0d7f",
}


def clean_studio_text(text: str, lang: str) -> str:
    """Normalize native Indic/Sanskrit text identically to `StudioCharTokenizer`."""
    lc = lang if lang in STUDIO_VOCABS else detect_script_and_lang(text, default_lang="sa")[1]
    t = unicodedata.normalize("NFC", str(text or "")).strip()
    if not t:
        return ""

    lo, hi = _SCRIPT_RANGES[lc]
    has_target_script = any(lo <= ord(ch) <= hi for ch in t)
    if not has_target_script:
        if lc == "sa":
            t = iast_to_devanagari(t)
        else:
            t = to_native_model_text(t, lang=lc)

    if lc == "sa":
        t = re.sub(r"[\u0951-\u0954\u1cd0-\u1cff]", "", t)
        t = t.replace("॥", ".").replace("।", ",")

    if lc == "kn":
        t = re.sub(r"\u0cbf\u0c83(?=[\s,.;:!?।॥]|$)", "ಿಹಿ", t)
        t = re.sub(r"\u0cc1\u0c83(?=[\s,.;:!?।॥]|$)", "ುಹು", t)
        t = re.sub(r"\u0c83(?=[\s,.;:!?।॥]|$)", "ಹ", t)
        t = t.replace("\u0c83", "ಹ್")
        t = t.replace("\u0c99\u0ccd\u0cae", "ಂಗ್ಮ")
        t = t.replace("\u0c99\u0ccd", "ಂ").replace("\u0c99", "ಂಗ")
        t = (
            t.replace("\u0cb1", "ರ")
            .replace("\u0cc4", "ೃ")
            .replace("\u0cde", "ಳ")
            .replace("\u0cd6", "ೈ")
            .replace("\u0ce0", "ಋ")
            .replace("\u0c8c", "ಲ")
            .replace("\u0ce1", "ಲ")
            .replace("\u0c81", "ಂ")
        )

    if lc == "ta":
        t = t.replace("\u0bd0", "ஓம்").replace("\u0bb6", "ஸ")

    if lc == "ml":
        for seq, atomic in _MALAYALAM_CHILLU_SEQ.items():
            t = t.replace(seq, atomic)
        t = (
            t.replace("\u0d4e", "\u0d7c")
            .replace("\u0d54", "ം")
            .replace("\u0d3a", "റ")
            .replace("\u0d44", "ൃ")
        )

    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = (
        t.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", " ")
        .replace("—", ", ")
        .replace("…", ".")
        .replace(";", ",")
        .replace(":", ",")
        .replace("-", " ")
    )
    if lc in ("kn", "ta", "te", "ml"):
        t = t.replace("॥", ".").replace("।", ".")
    elif lc == "hi":
        t = t.replace("॥", "।")

    t = t.lower()
    t = _AUX_SYMBOLS_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    if lc in ("ml", "ta") and t:
        if t.endswith(","):
            stem = re.sub(r"[,;\s]+$", "", t).strip()
            if stem:
                t = f"{stem} ,"
        else:
            stem = re.sub(r"[.,!?;\s]+$", "", t).strip()
            if stem:
                t = f"{stem} . ."
    return t


def studio_text_to_ids(text: str, lang: str | None = None) -> tuple[list[int], str, str]:
    """Return `(studio_ids, lang_code, cleaned_native_text)` with interleaved `<BLNK>` tokens."""
    _, resolved_lang = detect_script_and_lang(text, default_lang=lang)
    lc = resolved_lang if resolved_lang in STUDIO_VOCABS else "sa"
    spec = STUDIO_VOCABS[lc]
    c2i = spec["char_to_id"]
    blank_id = int(spec["blank_id"])

    cleaned = clean_studio_text(text, lc)
    raw_ids = [c2i[ch] for ch in cleaned if ch in c2i]
    if not raw_ids:
        raw_ids = [c2i.get(" ", int(spec["pad_id"]))]

    interleaved: list[int] = [blank_id]
    for cid in raw_ids:
        interleaved.append(int(cid))
        interleaved.append(blank_id)
    return interleaved, lc, cleaned
