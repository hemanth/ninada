"""Pan-Indic Akshara and Classical/Vedic Sanskrit Phoneme G2P front-end for Ninada."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Indices 0..72 align 1-to-1 with facebook/mms-tts-hin vocab.json for warm-starting;
# indices 73..86 add Dravidian and Classical/Vedic Sanskrit phoneme extensions.
VOCAB_TOKENS: list[str] = [
    "फ", "4", "1", "-", "अ", "इ", "8", " ", "0", "छ",
    "न", "ए", "ऐ", "़", "ष", "ि", "ँ", "च", "2", "ऑ",
    "थ", "भ", "ी", "\u200d", "ॅ", "3", "ञ", "ै", "ु", "ठ",
    "ं", "ॉ", "उ", "_", "ई", "ः", "ह", "ध", "ल", "र",
    "स", "ब", "ख", "ण", "'", "`", "व", "घ", "प", "ग",
    "ढ", "य", "े", "्", "ा", "आ", "ड", "ज", "झ", "श",
    "औ", "ो", "द", "ृ", "ौ", "ऊ", "ू", "ओ", "ट", "त",
    "क", "म", "<unk>",
    # Extended Pan-Indic & Classical/Vedic Sanskrit symbols (73..86)
    "\u0933", "\u0934", "\u0931", "\u0929", "\u090e", "\u0912", "\u0946", "\u094a",
    "\u0960", "\u0944", "\u0964", "\u0965", "\u090b", "\u0919",
]

TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN: dict[int, str] = {idx: tok for idx, tok in enumerate(VOCAB_TOKENS)}
BLANK_ID: int = 0
UNK_ID: int = TOKEN_TO_ID["<unk>"]

LANGUAGE_IDS: dict[str, int] = {
    "sa": 0,  # Sanskrit
    "hi": 1,  # Hindi
    "kn": 2,  # Kannada
    "ta": 3,  # Tamil
    "te": 4,  # Telugu
    "ml": 5,  # Malayalam
}
ID_TO_LANGUAGE: dict[int, str] = {v: k for k, v in LANGUAGE_IDS.items()}

# Articulatory warm-start donor characters in base mms-tts-hin (0..72) for extended tokens (73..86)
WARMSTART_DONOR_MAP: dict[str, str] = {
    "\u0933": "\u0932",
    "\u0934": "\u0930",
    "\u0931": "\u0930",
    "\u0929": "\u0928",
    "\u090e": "\u090f",
    "\u0912": "\u0913",
    "\u0946": "\u0947",
    "\u094a": "\u094b",
    "\u0960": "\u0943",
    "\u0944": "\u0943",
    "\u0964": "-",
    "\u0965": "-",
    "\u090b": "\u0943",
    "\u0919": "\u0923",
}

BRAHMIC_BASES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "odia"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
)

MALAYALAM_CHILLU_CP: dict[int, str] = {
    0x0D7A: "\u0923\u094d",
    0x0D7B: "\u0928\u094d",
    0x0D7C: "\u0930\u094d",
    0x0D7D: "\u0932\u094d",
    0x0D7E: "\u0933\u094d",
    0x0D7F: "\u0915\u094d",
}

DEV_CONSONANTS = set(chr(cp) for cp in range(0x0915, 0x093A)) | {"\u0931", "\u0934", "\u0929"}
DEV_VOWELS_INDEP = set(chr(cp) for cp in range(0x0904, 0x0915)) | {"\u0960", "\u090c"}
DEV_MATRAS = set(chr(cp) for cp in range(0x093E, 0x094D)) | {"\u0944", "\u0962", "\u0963"}
VIRAMA = "\u094d"
ANUSVARA = "\u0902"
VISARGA = "\u0903"
CHANDRABINDU = "\u0901"

IAST_CONSONANTS: list[tuple[str, str]] = [
    ("k\u1e63", "\u0915\u094d\u0937"), ("j\u00f1", "\u091c\u094d\u091e"),
    ("kh", "\u0916"), ("gh", "\u0918"), ("\u1e45", "\u0919"),
    ("ch", "\u091b"), ("jh", "\u091d"), ("\u00f1", "\u091e"),
    ("\u1e6dh", "\u0920"), ("\u1e0dh", "\u0922"), ("\u1e47", "\u0923"), ("\u1e6d", "\u091f"), ("\u1e0d", "\u0921"),
    ("th", "\u0925"), ("dh", "\u0927"),
    ("ph", "\u092b"), ("bh", "\u092d"),
    ("\u015b", "\u0936"), ("\u1e63", "\u0937"), ("\u1e3b", "\u0934"), ("\u1e37", "\u0933"), ("\u1e5f", "\u0931"), ("\u1e49", "\u0929"),
    ("k", "\u0915"), ("g", "\u0917"), ("c", "\u091a"), ("j", "\u091c"),
    ("t", "\u0924"), ("d", "\u0926"), ("n", "\u0928"),
    ("p", "\u092a"), ("b", "\u092c"), ("m", "\u092e"),
    ("y", "\u092f"), ("r", "\u0930"), ("l", "\u0932"), ("v", "\u0935"), ("w", "\u0935"),
    ("s", "\u0938"), ("h", "\u0939"),
]

IAST_VOWELS: list[tuple[str, str, str]] = [
    ("ai", "\u0910", "\u0948"), ("au", "\u0914", "\u094c"),
    ("\u0101", "\u0906", "\u093e"), ("\u012b", "\u0908", "\u0940"), ("\u016b", "\u090a", "\u0942"),
    ("\u1e5d", "\u0960", "\u0944"), ("\u1e5b", "\u090b", "\u0943"),
    ("\u0113", "\u090f", "\u0947"), ("\u014d", "\u0913", "\u094b"),
    ("a", "\u0905", ""), ("i", "\u0907", "\u093f"), ("u", "\u0909", "\u0941"),
    ("e", "\u090f", "\u0947"), ("o", "\u0913", "\u094b"),
]


def detect_script_and_lang(text: str, default_lang: str = "sa") -> tuple[str, str]:
    """Auto-detect dominant Brahmic/IAST script and likely Indic language code."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, sname in BRAHMIC_BASES:
            if lo <= cp <= hi:
                counts[sname] = counts.get(sname, 0) + 1
                break
    if not counts:
        if any(c in text.lower() for c in "\u0101\u012b\u016b\u1e5b\u1e5d\u1e37\u015b\u1e63\u1e6d\u1e0d\u1e47\u1e45\u00f1\u1e41\u1e25"):
            return "iast", default_lang
        return "latin", default_lang
    dom = max(counts.items(), key=lambda x: x[1])[0]
    script_to_lang = {
        "kannada": "kn",
        "tamil": "ta",
        "telugu": "te",
        "malayalam": "ml",
    }
    if dom in script_to_lang:
        return dom, script_to_lang[dom]
    return dom, default_lang


def iast_to_devanagari(text: str) -> str:
    """Convert ISO-15919 / IAST romanized Sanskrit/Indic text to Unicode Devanagari."""
    s = unicodedata.normalize("NFC", text.strip()).lower()
    s = re.sub(r"o\u1e41", "\u0913\u092e\u094d", s)
    s = re.sub(r"\bom\b", "\u0913\u092e\u094d", s)
    out: list[str] = []
    i = 0
    n = len(s)
    prev_was_consonant = False
    while i < n:
        if s[i].isspace():
            if prev_was_consonant:
                out.append(VIRAMA)
                prev_was_consonant = False
            out.append(" ")
            i += 1
            continue
        if s[i] in ("|", "/"):
            if prev_was_consonant:
                out.append(VIRAMA)
                prev_was_consonant = False
            if i + 1 < n and s[i + 1] == s[i]:
                out.append("\u0965")
                i += 2
            else:
                out.append("\u0964")
                i += 1
            continue
        if s[i] in ("'", "\u2019"):
            out.append("\u093d")
            prev_was_consonant = False
            i += 1
            continue
        if s[i] in ("\u1e41", "\u1e43"):
            out.append(ANUSVARA)
            prev_was_consonant = False
            i += 1
            continue
        if s[i] == "\u1e25":
            out.append(VISARGA)
            prev_was_consonant = False
            i += 1
            continue

        matched_v = False
        for rom, indep, matra in IAST_VOWELS:
            if s.startswith(rom, i):
                if prev_was_consonant:
                    if matra:
                        out.append(matra)
                else:
                    out.append(indep)
                prev_was_consonant = False
                i += len(rom)
                matched_v = True
                break
        if matched_v:
            continue

        matched_c = False
        for rom, dev in IAST_CONSONANTS:
            if s.startswith(rom, i):
                if prev_was_consonant:
                    out.append(VIRAMA)
                out.append(dev)
                prev_was_consonant = True
                i += len(rom)
                matched_c = True
                break
        if matched_c:
            continue

        if prev_was_consonant:
            out.append(VIRAMA)
            prev_was_consonant = False
        if s[i] in "-,.":
            out.append("-")
        i += 1

    if prev_was_consonant:
        out.append(VIRAMA)
    return "".join(out)


def brahmic_to_devanagari(text: str) -> str:
    """Map any Unicode Brahmic script (Kannada, Tamil, Telugu, Malayalam, Bengali, etc.) to Devanagari."""
    norm = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in norm:
        cp = ord(ch)
        if cp in MALAYALAM_CHILLU_CP:
            out.append(MALAYALAM_CHILLU_CP[cp])
            continue
        if 0x0900 <= cp <= 0x097F:
            out.append(ch)
            continue
        mapped = False
        for lo, hi, _ in BRAHMIC_BASES[1:]:
            if lo <= cp <= hi:
                offset = cp - lo
                if 0x01 <= offset <= 0x4D:
                    out.append(chr(0x0900 + offset))
                elif 0x66 <= offset <= 0x6F:
                    out.append(str(offset - 0x66))
                else:
                    out.append(" ")
                mapped = True
                break
        if not mapped:
            out.append(ch)
    return "".join(out)


def _homorganic_nasal(next_consonant: str) -> str:
    cp = ord(next_consonant)
    if 0x0915 <= cp <= 0x0919:
        return "\u0919\u094d"
    if 0x091A <= cp <= 0x091E:
        return "\u091e\u094d"
    if 0x091F <= cp <= 0x0923:
        return "\u0923\u094d"
    if 0x0924 <= cp <= 0x0928:
        return "\u0928\u094d"
    if 0x092A <= cp <= 0x092E:
        return "\u092e\u094d"
    return ANUSVARA


def apply_phonological_rules(dev_text: str, lang: str = "sa") -> str:
    """Apply language-specific Pan-Indic & Classical/Vedic Sanskrit phonological rules."""
    s = re.sub(r"[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]", "", dev_text)
    s = s.replace("\u1cf5", VISARGA).replace("\u1cf6", VISARGA)
    s = s.replace("\u0950", "\u0913\u092e\u094d")
    s = re.sub(r"([\u0947\u094b\u093e\u090f\u0913\u0906])\u093d+", r"\1", s)
    s = s.replace("\u093d", "\u0905")
    s = s.replace("\u0962", "\u093f").replace("\u0963", "\u0940").replace("\u090c", "\u0932\u0943")

    out_digits = []
    for ch in s:
        cp = ord(ch)
        if 0x0966 <= cp <= 0x096F:
            out_digits.append(str(cp - 0x0966))
        else:
            out_digits.append(ch)
    s = "".join(out_digits)

    if lang == "sa":
        chars = list(s)
        res: list[str] = []
        n = len(chars)
        for idx, ch in enumerate(chars):
            if ch == ANUSVARA and idx + 1 < n and chars[idx + 1] in DEV_CONSONANTS:
                nas = _homorganic_nasal(chars[idx + 1])
                if nas[0] in TOKEN_TO_ID:
                    res.append(nas)
                else:
                    res.append(ANUSVARA)
            else:
                res.append(ch)
        s = "".join(res)

    elif lang == "ta":
        pairs = [
            ("\u0919\u094d\u0915", "\u0919\u094d\u0917"), ("\u0902\u0915", "\u0902\u0917"),
            ("\u091e\u094d\u091b", "\u091e\u094d\u091d"), ("\u091e\u094d\u091a", "\u091e\u094d\u091c"), ("\u0902\u091a", "\u0902\u091c"),
            ("\u0923\u094d\u091f", "\u0923\u094d\u0921"), ("\u0902\u091f", "\u0902\u0921"),
            ("\u0928\u094d\u0924", "\u0928\u094d\u0926"), ("\u0929\u094d\u0924", "\u0928\u094d\u0926"), ("\u0902\u0924", "\u0902\u0926"),
            ("\u092e\u094d\u092a", "\u092e\u094d\u092c"), ("\u0902\u092a", "\u0902\u092c"),
        ]
        for src, dst in pairs:
            s = s.replace(src, dst)

    s = s.replace("||", "\u0965").replace("|", "\u0964")
    s = re.sub(r"[,;:!?]+", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def segment_aksharas(dev_text: str) -> list[dict[str, Any]]:
    """Segment canonical Devanagari into orthographic/prosodic aksharas with guru/laghu weight."""
    aksharas: list[dict[str, Any]] = []
    i = 0
    n = len(dev_text)
    LONG_VOWELS = {"\u0906", "\u0908", "\u090a", "\u0960", "\u090f", "\u0910", "\u0913", "\u0914"}
    LONG_MATRAS = {"\u093e", "\u0940", "\u0942", "\u0944", "\u0947", "\u0948", "\u094b", "\u094c"}

    while i < n:
        ch = dev_text[i]
        if ch.isspace() or ch in ("-", "\u0964", "\u0965"):
            i += 1
            continue
        if ch in DEV_VOWELS_INDEP:
            buf = ch
            i += 1
            is_long = ch in LONG_VOWELS
            while i < n and dev_text[i] in (ANUSVARA, VISARGA, CHANDRABINDU):
                if dev_text[i] in (ANUSVARA, VISARGA):
                    is_long = True
                buf += dev_text[i]
                i += 1
            aksharas.append({"akshara": buf, "weight": "G" if is_long else "L"})
            continue
        if ch in DEV_CONSONANTS:
            buf = ch
            i += 1
            while i + 1 < n and dev_text[i] == VIRAMA and dev_text[i + 1] in DEV_CONSONANTS:
                buf += dev_text[i] + dev_text[i + 1]
                i += 2
            is_long = False
            if i < n and dev_text[i] == VIRAMA:
                buf += VIRAMA
                i += 1
                if aksharas:
                    aksharas[-1]["weight"] = "G"
                aksharas.append({"akshara": buf, "weight": "C"})
                continue
            if i < n and dev_text[i] in DEV_MATRAS:
                if dev_text[i] in LONG_MATRAS:
                    is_long = True
                buf += dev_text[i]
                i += 1
            while i < n and dev_text[i] in (ANUSVARA, VISARGA, CHANDRABINDU):
                if dev_text[i] in (ANUSVARA, VISARGA):
                    is_long = True
                buf += dev_text[i]
                i += 1
            j = i
            while j < n and dev_text[j].isspace():
                j += 1
            if j + 1 < n and dev_text[j] in DEV_CONSONANTS and dev_text[j + 1] == VIRAMA:
                is_long = True
            aksharas.append({"akshara": buf, "weight": "G" if is_long else "L"})
            continue
        i += 1
    return aksharas


def text_to_ids(
    text: str,
    lang: str | None = None,
    add_blank: bool = True,
    min_len: int = 8,
) -> dict[str, Any]:
    """Convert any Indic/Sanskrit/IAST string into blank-interleaved VITS input_ids + metadata."""
    detected_script, detected_lang = detect_script_and_lang(text, default_lang=lang or "sa")
    active_lang = lang if lang in LANGUAGE_IDS else detected_lang

    if detected_script in ("iast", "latin"):
        dev = iast_to_devanagari(text)
    else:
        dev = brahmic_to_devanagari(text)

    phonemic = apply_phonological_rules(dev, lang=active_lang)
    raw_ids: list[int] = []
    tokens: list[str] = []
    for ch in phonemic:
        if ch in TOKEN_TO_ID:
            raw_ids.append(TOKEN_TO_ID[ch])
            tokens.append(ch)

    if not raw_ids:
        raw_ids = [TOKEN_TO_ID["\u0905"]]
        tokens = ["\u0905"]

    if add_blank:
        interleaved: list[int] = [BLANK_ID]
        for tid in raw_ids:
            interleaved.append(tid)
            interleaved.append(BLANK_ID)
        input_ids = interleaved
    else:
        input_ids = raw_ids

    while len(input_ids) < min_len:
        input_ids.append(BLANK_ID)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "lang": active_lang,
        "lang_id": LANGUAGE_IDS[active_lang],
        "script": detected_script,
        "normalized_devanagari": phonemic,
        "tokens": tokens,
        "aksharas": segment_aksharas(phonemic),
    }
