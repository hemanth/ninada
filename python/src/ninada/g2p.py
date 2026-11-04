"""Pan-Indic Akshara, Per-Language Native Brahmic, and Classical/Vedic Sanskrit Phoneme G2P for Ninada."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .ipa_g2p import IPA_ID_TO_TOKEN, IPA_VOCAB, RETROFLEX_LATERAL_ID, ipa_to_ids, text_to_ipa

# Canonical 87-symbol Pan-Indic & Classical/Vedic Sanskrit inspection vocabulary
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

# Dedicated non-colliding padding / blank token definitions
PAD_TOKEN: str = "_"
BLANK_ID: int = TOKEN_TO_ID.get(PAD_TOKEN, 33)
PAD_TOKEN_ID: dict[str, int] = {
    "sa": 49,
    "kn": 49,
    "hi": 33,
    "ta": 4,
    "te": 45,
    "ml": 43,
    "en": 0,
    "mr": 33,
    "bn": 0,
    "gu": 0,
    "pa": 0,
    "or": 0,
}
VIRAMA_TOKEN_IDS: dict[str, int] = {
    "sa": 15,  # ್ (\u0ccd)
    "kn": 15,  # ್ (\u0ccd)
    "hi": 53,  # ् (\u094d)
    "ta": 18,  # ் (\u0bcd)
    "te": 48,  # ్ (\u0c4d)
    "ml": 36,  # ് (\u0d4d)
    "en": -1,
    "mr": 53,  # ् (\u094d)
    "bn": 56,  # ् (\u09cd)
    "gu": 60,  # ् (\u0acd)
    "pa": 53,  # ् (\u0a4d)
    "or": 58,  # ् (\u0b4d)
}
UNK_ID: int = TOKEN_TO_ID["<unk>"]

LANGUAGE_IDS: dict[str, int] = {
    "sa": 0,   # Sanskrit (Classical & Vedic, zero-schwa-deletion Kannada-isomorphic phoneme map)
    "hi": 1,   # Hindi (Native Devanagari)
    "kn": 2,   # Kannada (Native Kannada)
    "ta": 3,   # Tamil (Native Tamil)
    "te": 4,   # Telugu (Native Telugu)
    "ml": 5,   # Malayalam (Native Malayalam with atomic chillu letters)
    "en": 6,   # English (Indian English Studio)
    "mr": 7,   # Marathi (Native Devanagari)
    "bn": 8,   # Bengali (Native Eastern Nagari)
    "gu": 9,   # Gujarati (Native Gujarati)
    "pa": 10,  # Punjabi (Native Gurmukhi)
    "or": 11,  # Odia (Native Odia)
}
ID_TO_LANGUAGE: dict[int, str] = {v: k for k, v in LANGUAGE_IDS.items()}

class NativeVocabList(list):
    """Token list conforming strictly to frozen embedding matrix while supporting donor phoneme membership."""

    def __init__(self, items: list[str], donor_chars: tuple[str, ...] = ()):
        super().__init__(items)
        self._donor_chars = set(donor_chars)

    def __contains__(self, item: object) -> bool:
        return super().__contains__(item) or item in self._donor_chars


# Exact per-language native specialist vocabularies (blank_id == 0 across all 6 languages)
NATIVE_VOCABS: dict[str, list[str]] = {
    "kn": NativeVocabList([
        "'", "ಓ", "ಮ", "ಆ", "ಭ", "ಋ", "ಚ", "ಟ", "ು", "ೇ",
        "ಜ", "ಏ", "ಾ", "ಪ", "ಶ", "್", "ಈ", "ಔ", "ಞ", "1",
        "ನ", "ಎ", "ಸ", "ವ", "ಇ", "ಘ", "5", "ಊ", "ಳ", "ಖ",
        "ೃ", "ೆ", "-", "ತ", "ಝ", "ಢ", " ", "ಅ", "ಧ", "ೌ",
        "ಹ", "ಯ", "9", "ಫ", "ಛ", "8", "6", "ಲ", "ಣ", "_",
        "ದ", "ಬ", "ಠ", "ಡ", "ಃ", "3", "ೋ", "ಂ", "ಒ", "ೀ",
        "ಕ", "ಿ", "ಥ", "ರ", "ೂ", "ಐ", "ೈ", "ಉ", "2", "0",
        "ಗ", "ೊ", "ಷ", "4", "7", "<unk>",
    ], donor_chars=("\u0c99",)),  # 76 tokens (0..75) conforming to checkpoint, ಙ supported via donor
    "hi": [
        "फ", "4", "1", "-", "अ", "इ", "8", " ", "0", "छ",
        "न", "ए", "ऐ", "़", "ष", "ि", "ँ", "च", "2", "ऑ",
        "थ", "भ", "ी", "\u200d", "ॅ", "3", "ञ", "ै", "ु", "ठ",
        "ं", "ॉ", "उ", "_", "ई", "ः", "ह", "ध", "ल", "र",
        "स", "ब", "ख", "ण", "'", "`", "व", "घ", "प", "ग",
        "ढ", "य", "े", "्", "ा", "आ", "ड", "ज", "झ", "श",
        "औ", "ो", "द", "ृ", "ौ", "ऊ", "ू", "ओ", "ट", "त",
        "क", "म", "<unk>",
    ],
    "ta": [
        "3", "அ", "ஈ", "உ", "_", "ந", "க", " ", "ே", "ா",
        "ெ", "ஊ", "ீ", "'", "7", "a", "ஏ", "வ", "்", "ஹ",
        "ூ", "ல", "ட", "2", "ழ", "ர", "ம", "ஒ", "ற", "ஞ",
        "ச", "9", "ி", "ஸ", "ை", "ன", "6", "எ", "இ", "ய",
        "ௌ", "த", "ோ", "ள", "1", "ஆ", "ப", "0", "ண", "ஓ",
        "ங", "ு", "ஐ", "ஜ", "4", "ஷ", "ொ", "5", "<unk>",
    ],
    "te": NativeVocabList([
        "త", "ఞ", "భ", "థ", "ర", "న", "ో", "ఘ", "'", "ఐ",
        "గ", "ే", "జ", "ఔ", "హ", "ా", "డ", "మ", "స", " ",
        "క", "ఒ", "ూ", "వ", "ధ", "శ", "ౌ", "ప", "ః", "ఠ",
        "ఖ", "ష", "–", "ణ", "ఫ", "ి", "ొ", "ఏ", "ఎ", "ృ",
        "ఓ", "య", "అ", "ీ", "ె", "_", "చ", "ు", "్", "ద",
        "ల", "ఉ", "ఢ", "-", "బ", "ఈ", "ట", "ఛ", "ళ", "ఊ",
        "6", "ం", "ఇ", "ఆ", "ై", "<unk>",
    ], donor_chars=("\u0c1d",)),  # 66 tokens (0..65) conforming to checkpoint, ఝ supported via donor
    "ml": NativeVocabList([
        "—", "ത", "ര", "ള", "ഴ", "ഒ", "0", "ൂ", "ഖ", "ഞ",
        "ഇ", "ഡ", "ി", "ഊ", "ച", "ാ", "യ", "ഷ", "i", "ഉ",
        "ഫ", "ല", "ഠ", "4", "െ", "ൗ", "ആ", "സ", "ു", "ം",
        "m", "ജ", "ധ", "-", "ഹ", "ഛ", "്", "ട", "ഃ", "ഗ",
        "ണ", "ൽ", "ോ", "_", "റ", "ഏ", "6", "എ", "ർ", "ഥ",
        "ഘ", "ഓ", "ീ", "ൈ", "2", "േ", "ക", "1", "ൾ", "ങ",
        "ഭ", " ", "ഢ", "അ", "മ", "ബ", "3", "ഈ", "ഔ", "പ",
        "ൺ", "ന", "ൊ", "o", "ൃ", "c", "വ", "ൻ", "ദ", "5",
        "ഐ", "'", "q", "ശ", "<unk>",
    ], donor_chars=("\u0d1d",)),  # 85 tokens (0..84) conforming to checkpoint, ഝ supported via donor
}
# Sanskrit uses the full zero-schwa-deletion Kannada phoneme inventory (0..75) + 3 Vedic/Classical extensions (76..78)
NATIVE_VOCABS["sa"] = list(NATIVE_VOCABS["kn"]) + ["\u0c99", "\u0ce0", "\u0cc4"]
NATIVE_VOCABS["en"] = ["_"] + list(" abcdefghijklmnopqrstuvwxyz'-,.?!") + ["<unk>"]
NATIVE_VOCABS["mr"] = list(NATIVE_VOCABS["hi"][:-1]) + ["\u0933", "<unk>"]
NATIVE_VOCABS["bn"] = (
    ["_"]
    + list(" ',.-?")
    + sorted(chr(cp) for cp in range(0x0981, 0x09FA) if unicodedata.category(chr(cp))[0] in ("L", "M", "N"))
    + ["<unk>"]
)
NATIVE_VOCABS["gu"] = (
    ["_"]
    + list(" ',.-?")
    + sorted(chr(cp) for cp in range(0x0A81, 0x0AF2) if unicodedata.category(chr(cp))[0] in ("L", "M", "N"))
    + ["<unk>"]
)
NATIVE_VOCABS["pa"] = (
    ["_"]
    + list(" ',.-?")
    + sorted(chr(cp) for cp in range(0x0A01, 0x0A76) if unicodedata.category(chr(cp))[0] in ("L", "M", "N"))
    + ["<unk>"]
)
NATIVE_VOCABS["or"] = (
    ["_"]
    + list(" ',.-?")
    + sorted(chr(cp) for cp in range(0x0B01, 0x0B73) if unicodedata.category(chr(cp))[0] in ("L", "M", "N"))
    + ["<unk>"]
)

for _lc_v in ("bn", "gu", "pa", "or"):
    _vir_c = {"bn": "\u09cd", "gu": "\u0acd", "pa": "\u0a4d", "or": "\u0b4d"}[_lc_v]
    if _vir_c in NATIVE_VOCABS[_lc_v]:
        VIRAMA_TOKEN_IDS[_lc_v] = NATIVE_VOCABS[_lc_v].index(_vir_c)

NATIVE_TOKEN_TO_ID: dict[str, dict[str, int]] = {
    lc: {tok: idx for idx, tok in enumerate(toks)} for lc, toks in NATIVE_VOCABS.items()
}

NATIVE_BASE_CHECKPOINTS: dict[str, str] = {
    "sa": "facebook/mms-tts-kan",
    "kn": "facebook/mms-tts-kan",
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "ml": "facebook/mms-tts-mal",
    "en": "facebook/mms-tts-eng",
    "mr": "facebook/mms-tts-mar",
    "bn": "facebook/mms-tts-ben",
    "gu": "facebook/mms-tts-guj",
    "pa": "facebook/mms-tts-pan",
    "or": "facebook/mms-tts-ory",
}

# Articulatory donor mappings for base MMS checkpoints
NATIVE_DONOR_MAP: dict[str, dict[str, str]] = {
    "sa": {
        "\u0c99": "\u0ca3",  # ಙ (ṅa) <- ಣ (ṇa)
        "\u0ce0": "\u0c8b",  # ೠ (ṝ)  <- ಋ (ṛ)
        "\u0cc4": "\u0cc3",  # ೄ (ṝ)  <- ೃ (ṛ)
    },
    "kn": {
        "\u0c99": "\u0ca8",  # ಙ (ṅa) <- ನ (na)
    },
    "te": {
        "\u0c1d": "\u0c1c",  # ఝ (jha) <- జ (ja)
    },
    "ml": {
        "\u0d1d": "\u0d1c",  # ഝ (jha) <- ജ (ja)
    },
}

# Articulatory donor characters for Sanskrit's 3 extended tokens (76..78) on facebook/mms-tts-kan
SANSKRIT_DONOR_MAP: dict[str, str] = NATIVE_DONOR_MAP["sa"]

# Kept for backward compatibility
WARMSTART_DONOR_MAP: dict[str, str] = dict(SANSKRIT_DONOR_MAP)

VEDIC_SVARA_MARKS: dict[str, str] = {
    "\u0951": "udatta",
    "\u0952": "anudatta",
    "\u0953": "svarita",
    "\u0954": "svarita",
    "\u1cd4": "udatta",
    "\u1cda": "udatta",
    "\u1cdb": "udatta",
    "\u1cdc": "anudatta",
    "\u1cdd": "svarita",
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

MALAYALAM_ZWJ_CHILLU: list[tuple[str, str]] = [
    ("\u0d23\u0d4d\u200d", "\u0d7a"),
    ("\u0d28\u0d4d\u200d", "\u0d7b"),
    ("\u0d30\u0d4d\u200d", "\u0d7c"),
    ("\u0d32\u0d4d\u200d", "\u0d7d"),
    ("\u0d33\u0d4d\u200d", "\u0d7e"),
]

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
    """Auto-detect dominant Brahmic/IAST/Latin script and likely Indic/English language code."""
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
        "bengali": "bn",
        "gurmukhi": "pa",
        "gujarati": "gu",
        "odia": "or",
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
        elif s[i].isdigit():
            out.append(s[i])
        i += 1

    if prev_was_consonant:
        out.append(VIRAMA)
    return "".join(out)


_EN_DIGIT_DEV: dict[str, str] = {
    "0": "जीरो",
    "1": "वन",
    "2": "टू",
    "3": "थ्री",
    "4": "फोर",
    "5": "फाइव",
    "6": "सिक्स",
    "7": "सेवन",
    "8": "एट",
    "9": "नाइन",
}

_EN_DIGIT_WORDS: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_PA_DIGIT_DEV: dict[str, str] = {
    "0": "सिफ़र",
    "1": "इक्क",
    "2": "दो",
    "3": "तिन्न",
    "4": "चार",
    "5": "पंज",
    "6": "छे",
    "7": "सत्त",
    "8": "अट्ਠ",
    "9": "नौं",
}


def _english_to_devanagari(text: str) -> str:
    """Convert English text into phonetic Devanagari aksharas for Tier-2 Klatt & prosodic inspection."""
    s = unicodedata.normalize("NFC", text.strip())
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = s.lower()
    for d, w in _EN_DIGIT_WORDS.items():
        s = re.sub(rf"\b{d}\b", f" {w} ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = (
        s.replace("tion", "shan")
        .replace("sion", "zhan")
        .replace("qu", "kw")
        .replace("ck", "k")
        .replace("wh", "w")
    )
    en_vows: list[tuple[str, str, str]] = [
        ("oo", "\u090a", "\u0942"),
        ("ee", "\u0908", "\u0940"),
        ("ea", "\u0908", "\u0940"),
        ("ai", "\u090f", "\u0947"),
        ("ay", "\u090f", "\u0947"),
        ("ou", "\u0914", "\u094c"),
        ("ow", "\u0913", "\u094b"),
        ("oa", "\u0913", "\u094b"),
        ("oi", "\u0913\u0907", "\u094b\u0907"),
        ("oy", "\u0913\u0907", "\u094b\u0907"),
        ("a", "\u0905", ""),
        ("e", "\u090f", "\u0947"),
        ("i", "\u0907", "\u093f"),
        ("o", "\u0913", "\u094b"),
        ("u", "\u0905", ""),
    ]
    en_cons: list[tuple[str, str]] = [
        ("tch", "\u091a"),
        ("sh", "\u0936"),
        ("ch", "\u091a"),
        ("th", "\u0925"),
        ("ph", "\u092b"),
        ("gh", "\u0917"),
        ("ng", "\u0919"),
        ("zh", "\u091d"),
        ("kh", "\u0916"),
        ("bh", "\u092d"),
        ("dh", "\u0927"),
        ("b", "\u092c"),
        ("d", "\u0921"),
        ("f", "\u092b"),
        ("g", "\u0917"),
        ("h", "\u0939"),
        ("j", "\u091c"),
        ("k", "\u0915"),
        ("l", "\u0932"),
        ("m", "\u092e"),
        ("n", "\u0928"),
        ("p", "\u092a"),
        ("q", "\u0915"),
        ("r", "\u0930"),
        ("s", "\u0938"),
        ("t", "\u091f"),
        ("v", "\u0935"),
        ("w", "\u0935"),
        ("x", "\u0915\u094d\u0938"),
        ("z", "\u091c"),
    ]
    out: list[str] = []
    i = 0
    n = len(s)
    prev_c = False
    while i < n:
        ch = s[i]
        if ch.isspace():
            if prev_c:
                out.append(VIRAMA)
                prev_c = False
            out.append(" ")
            i += 1
            continue
        if ch in _EN_DIGIT_DEV:
            if prev_c:
                out.append(VIRAMA)
                prev_c = False
            out.append(_EN_DIGIT_DEV[ch])
            i += 1
            continue
        if ch == "c":
            nxt = s[i + 1] if i + 1 < n else ""
            if prev_c:
                out.append(VIRAMA)
            out.append("\u0938" if nxt in ("e", "i", "y") else "\u0915")
            prev_c = True
            i += 1
            continue
        if ch == "y":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in "aeiou":
                if prev_c:
                    out.append(VIRAMA)
                out.append("\u092f")
                prev_c = True
            else:
                out.append("\u0940" if prev_c else "\u0908")
                prev_c = False
            i += 1
            continue
        mv = False
        for rom, indep, matra in en_vows:
            if s.startswith(rom, i):
                if rom == "e" and prev_c and len(out) >= 2 and (i + 1 == n or s[i + 1].isspace() or s[i + 1] in ",.?!"):
                    # Silent word-final 'e' after consonant
                    out.append(VIRAMA)
                    prev_c = False
                    i += 1
                    mv = True
                    break
                if prev_c:
                    if matra:
                        out.append(matra)
                else:
                    out.append(indep)
                prev_c = False
                i += len(rom)
                mv = True
                break
        if mv:
            continue
        mc = False
        for rom, dev in en_cons:
            if s.startswith(rom, i):
                if prev_c:
                    out.append(VIRAMA)
                out.append(dev)
                prev_c = True
                i += len(rom)
                mc = True
                break
        if mc:
            continue
        if prev_c and ch not in ("'", "\u2019"):
            out.append(VIRAMA)
            prev_c = False
        i += 1
    if prev_c:
        out.append(VIRAMA)
    return "".join(out)


def brahmic_to_devanagari(text: str) -> str:
    """Map any Unicode Brahmic script (Kannada, Tamil, Telugu, Malayalam, Bengali, Gurmukhi, Gujarati, Odia) to Devanagari."""
    norm = unicodedata.normalize("NFC", text)
    for zwj_seq, atomic in MALAYALAM_ZWJ_CHILLU:
        norm = norm.replace(zwj_seq, atomic)
    norm = (
        norm.replace("\u0d4c", "\u0d57")
        .replace("\u0a74", "\u0907\u0915 \u0913\u0902\u0915\u093e\u0930")
        .replace("\u09af\u09bc", "\u09df")
    )
    # Punjabi Adhak (U+0A71): double the following consonant rather than stripping
    norm = re.sub(r"\u0a71([\u0a15-\u0a39\u0a59-\u0a5e]\u0a3c?)", r"\g<1>" + "\u0a4d" + r"\g<1>", norm)
    norm = norm.replace("\u0a71", "")
    special_cp: dict[int, str] = {
        0x09CE: "\u0924\u094d",  # Bengali ৎ -> त्
        0x09D7: "\u094c",        # Bengali ৗ -> ौ
        0x09DC: "\u0921\u093c",  # Bengali ড় -> ड़
        0x09DD: "\u0922\u093c",  # Bengali ঢ় -> ढ़
        0x09DF: "\u092f",        # Bengali য় -> य
        0x09F0: "\u0930",        # Assamese/Bengali ৰ -> र
        0x09F1: "\u0935",        # Assamese/Bengali ৱ -> व
        0x0A5C: "\u0921\u093c",  # Gurmukhi ੜ -> ड़
        0x0A70: "\u0902",        # Gurmukhi ੰ (Tippi) -> ं
        0x0A72: "\u0907",        # Gurmukhi ੲ -> इ
        0x0A73: "\u0909",        # Gurmukhi ੳ -> उ
        0x0B56: "\u0948",        # Odia ୖ -> ै
        0x0B57: "\u094c",        # Odia ୗ -> ौ
        0x0B5C: "\u0921\u093c",  # Odia ଡ଼ -> ड़
        0x0B5D: "\u0922\u093c",  # Odia ଢ଼ -> ढ़
        0x0B5F: "\u092f",        # Odia ୟ -> य
        0x0B71: "\u0935",        # Odia ୱ -> व
    }
    out: list[str] = []
    for ch in norm:
        cp = ord(ch)
        if cp in MALAYALAM_CHILLU_CP:
            out.append(MALAYALAM_CHILLU_CP[cp])
            continue
        if cp == 0x0D57:
            out.append("\u094c")
            continue
        if cp in special_cp:
            out.append(special_cp[cp])
            continue
        if 0x0900 <= cp <= 0x097F:
            out.append(ch)
            continue
        mapped = False
        for lo, hi, _ in BRAHMIC_BASES[1:]:
            if lo <= cp <= hi:
                offset = cp - lo
                if 0x01 <= offset <= 0x4D or offset in (0x50, 0x60, 0x62, 0x63):
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


def delete_hindi_schwa_word(word: str) -> str:
    """Apply rule-based word-final and medial VC_CV schwa deletion to a single Hindi word."""
    m_lead = re.match(r"^[^\w\u0900-\u097f]+", word)
    lead = m_lead.group(0) if m_lead else ""
    m_trail = re.search(r"[^\w\u0900-\u097f]+$", word)
    trail = m_trail.group(0) if m_trail else ""
    core = word[len(lead):len(word) - len(trail)] if (lead or trail) else word
    if not core:
        return word

    chars = list(core)
    n = len(chars)
    tokens: list[dict[str, Any]] = []
    i = 0
    while i < n:
        c = chars[i]
        has_nukta = False
        if i + 1 < n and chars[i + 1] == "\u093c":
            has_nukta = True

        is_c = c in DEV_CONSONANTS
        is_v = c in DEV_VOWELS_INDEP
        is_m = c in DEV_MATRAS
        is_vir = (c == VIRAMA)
        tokens.append({
            "char": c,
            "nukta": has_nukta,
            "is_c": is_c,
            "is_v": is_v,
            "is_m": is_m,
            "is_vir": is_vir,
            "has_schwa": False,
            "delete_schwa": False,
        })
        i += 1
        if has_nukta:
            i += 1

    num_tok = len(tokens)
    for idx in range(num_tok):
        t = tokens[idx]
        if t["is_c"]:
            if idx + 1 < num_tok:
                next_t = tokens[idx + 1]
                if next_t["is_vir"] or next_t["is_m"]:
                    t["has_schwa"] = False
                    continue
            t["has_schwa"] = True

    # Single-consonant monosyllables (e.g. न, व, छ) retain schwa
    total_nuclei = sum(1 for t in tokens if t["is_v"] or t["is_m"] or t["has_schwa"])
    if total_nuclei <= 1:
        return word

    # 1. Word-Final Schwa Deletion (with Sanskrit Tatsama / final cluster protection)
    last_c_idx = None
    for idx in range(num_tok - 1, -1, -1):
        if tokens[idx]["is_c"]:
            last_c_idx = idx
            break
        elif tokens[idx]["is_v"] or tokens[idx]["is_m"]:
            break

    if last_c_idx is not None and tokens[last_c_idx]["has_schwa"]:
        # Block deletion if preceded by virama (final conjunct like सत्य, मित्र, कष्ट, धर्म)
        if not (last_c_idx > 0 and tokens[last_c_idx - 1]["is_vir"]):
            tokens[last_c_idx]["has_schwa"] = False
            tokens[last_c_idx]["delete_schwa"] = True

    # 2. Medial Schwa Deletion (VC_CV evaluated right-to-left)
    first_nucleus_idx = None
    for idx in range(num_tok):
        if tokens[idx]["is_v"] or tokens[idx]["is_m"] or tokens[idx]["has_schwa"]:
            first_nucleus_idx = idx
            break

    for idx in range(num_tok - 1, -1, -1):
        t = tokens[idx]
        if not t["has_schwa"]:
            continue
        if t["char"] == "\u0939":  # 'ह' resists schwa deletion
            continue
        if first_nucleus_idx is not None and idx <= first_nucleus_idx:
            continue  # Initial syllable schwa never deletes

        # Consonant cluster context checks
        if idx > 0 and tokens[idx - 1]["is_vir"]:
            continue
        if idx >= 2 and tokens[idx - 2]["is_vir"]:
            continue
        if idx + 2 < num_tok and tokens[idx + 2]["is_vir"]:
            continue

        # Check preceding vowel nucleus
        has_prec_v = False
        for p in range(idx - 1, -1, -1):
            if tokens[p]["is_v"] or tokens[p]["is_m"] or tokens[p]["has_schwa"]:
                has_prec_v = True
                break
        if not has_prec_v:
            continue

        # Check following consonant with overt vowel (matra or independent vowel)
        has_foll_cv = False
        foll_c = None
        for f in range(idx + 1, num_tok):
            if tokens[f]["is_c"]:
                foll_c = f
                break
        if foll_c is not None:
            if foll_c + 1 < num_tok and (tokens[foll_c + 1]["is_m"] or tokens[foll_c + 1]["is_v"]):
                has_foll_cv = True

        if has_foll_cv:
            t["has_schwa"] = False
            t["delete_schwa"] = True

    out: list[str] = []
    for t in tokens:
        out.append(t["char"])
        if t["nukta"]:
            out.append("\u093c")
        if t["delete_schwa"]:
            out.append(VIRAMA)
    return lead + "".join(out) + trail


def delete_hindi_schwa(text: str) -> str:
    """Apply deterministic schwa deletion across all whitespace-delimited words in Hindi text."""
    words = text.split(" ")
    processed = [delete_hindi_schwa_word(w) for w in words]
    return " ".join(processed)


def apply_phonological_rules(dev_text: str, lang: str = "sa") -> str:
    """Apply language-specific Pan-Indic & Classical/Vedic Sanskrit phonological rules."""
    s = unicodedata.normalize("NFC", dev_text)
    s = re.sub(r"[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]", "", s)
    s = s.replace("\u1cf5", VISARGA).replace("\u1cf6", VISARGA)
    s = s.replace("\u0950", "\u0913\u092e\u094d")
    # Correct avagraha (ऽ) pūrvarūpa sandhi: absorb into preceding vowel (with optional space);
    # eliminate spurious insertion of independent 'अ' which breaks sandhi
    s = re.sub(r"([\u0905-\u0914\u093e-\u094c\u0915-\u0939])\s*\u093d+", r"\1", s)
    s = re.sub(r"\u093d+", "", s)
    s = s.replace("\u0962", "\u0943").replace("\u0963", "\u0944").replace("\u090c", "\u0932\u0943")

    out_digits = []
    for ch in s:
        cp = ord(ch)
        if 0x0966 <= cp <= 0x096F:
            out_digits.append(str(cp - 0x0966))
        else:
            out_digits.append(ch)
    s = "".join(out_digits)

    if lang in ("hi", "mr", "gu", "pa"):
        s = delete_hindi_schwa(s)
    elif lang == "sa":
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
    # Replace punctuation with clean prosodic whitespace boundaries instead of click-inducing hyphens
    s = re.sub(r"[,;:!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _devanagari_to_target_script(dev_text: str, target_lang: str) -> str:
    """Transliterate canonical Devanagari into target native Brahmic/Latin script when input was IAST or cross-script."""
    if target_lang in ("hi", "mr"):
        return dev_text.replace("\u0964", " ").replace("\u0965", " ")
    if target_lang == "en":
        # Romanize Devanagari fallback cleanly into lowercase ASCII for English specialist
        rev_map = {
            "\u0905": "a", "\u0906": "aa", "\u0907": "i", "\u0908": "ee", "\u0909": "u", "\u090a": "oo",
            "\u090f": "e", "\u0910": "ai", "\u0913": "o", "\u0914": "au",
            "\u093e": "aa", "\u093f": "i", "\u0940": "ee", "\u0941": "u", "\u0942": "oo",
            "\u0947": "e", "\u0948": "ai", "\u094b": "o", "\u094c": "au",
            "\u0902": "m", "\u0901": "n", "\u0903": "h",
            "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh", "\u0919": "ng",
            "\u091a": "ch", "\u091b": "chh", "\u091c": "j", "\u091d": "jh", "\u091e": "ny",
            "\u091f": "t", "\u0920": "th", "\u0921": "d", "\u0922": "dh", "\u0923": "n",
            "\u0924": "t", "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
            "\u092a": "p", "\u092b": "f", "\u092c": "b", "\u092d": "bh", "\u092e": "m",
            "\u092f": "y", "\u0930": "r", "\u0932": "l", "\u0933": "l", "\u0935": "v",
            "\u0936": "sh", "\u0937": "sh", "\u0938": "s", "\u0939": "h",
        }
        rom_out: list[str] = []
        n_d = len(dev_text)
        for idx, ch in enumerate(dev_text):
            if ch in ("\u094d", "\u093c"):
                continue
            if ch in ("\u0964", "\u0965"):
                rom_out.append(" ")
                continue
            rom_out.append(rev_map.get(ch, ch))
            if ch in DEV_CONSONANTS:
                nxt = dev_text[idx + 1] if idx + 1 < n_d else ""
                if nxt not in DEV_MATRAS and nxt != VIRAMA:
                    rom_out.append("a")
        return "".join(rom_out).lower()

    base_map = {
        "sa": 0x0C80,  # Kannada-isomorphic zero-schwa-deletion Sanskrit
        "kn": 0x0C80,
        "ta": 0x0B80,
        "te": 0x0C00,
        "ml": 0x0D00,
        "bn": 0x0980,
        "pa": 0x0A00,
        "gu": 0x0A80,
        "or": 0x0B00,
    }
    target_base = base_map.get(target_lang, 0x0C80)
    s = dev_text.replace("\u0901", "\u0902")
    if target_lang in ("bn", "or"):
        # Bengali and Odia have no native U+0x35 ('व'); map to U+0x3C/0x2C ('ব'/'ବ')
        s = s.replace("\u0935", "\u092c")
    elif target_lang == "pa":
        # Gurmukhi lacks ष (0x37) -> श (0x36), ऋ -> रि
        s = s.replace("\u0937", "\u0936").replace("\u090b", "\u0930\u093f").replace("\u0943", "\u0930\u093f")

    if target_lang == "ta":
        # Fold non-Tamil aspirated/voiced stops & vocalic-r into standard Tamil orthography
        ta_fold = {
            "\u0916": "\u0915", "\u0917": "\u0915", "\u0918": "\u0915",
            "\u091b": "\u091a", "\u091d": "\u091c",
            "\u0920": "\u091f", "\u0921": "\u091f", "\u0922": "\u091f",
            "\u0925": "\u0924", "\u0926": "\u0924", "\u0927": "\u0924",
            "\u092b": "\u092a", "\u092c": "\u092a", "\u092d": "\u092a",
            "\u0936": "\u0938", "\u090b": "\u0930\u093f", "\u0943": "\u0930\u093f",
            "\u0902": "\u092e\u094d", "\u0903": "",
        }
        s = "".join(ta_fold.get(c, c) for c in s)

    out: list[str] = []
    for ch in s:
        if ch in ("\u0964", "\u0965"):
            out.append(" ")
            continue
        cp = ord(ch)
        if cp == 0x093C:
            if target_lang in ("bn", "pa", "or"):
                out.append(chr(target_base + 0x3C))
            continue  # strip Devanagari nukta in Southern/Gujarati Brahmic targets
        if 0x0901 <= cp <= 0x0963:
            offset = cp - 0x0900
            target_ch = chr(target_base + offset)
            if unicodedata.category(target_ch)[0] in ("L", "M"):
                out.append(target_ch)
        else:
            out.append(ch)
    return "".join(out)


def to_native_model_text(text: str, lang: str, detected_script: str, phonemic_dev: str) -> str:
    """Produce normalized native-script text matching the target specialist model's exact vocabulary."""
    s_raw = unicodedata.normalize("NFC", text)
    script_for_lang = {
        "kn": "kannada",
        "hi": "devanagari",
        "mr": "devanagari",
        "ta": "tamil",
        "te": "telugu",
        "ml": "malayalam",
        "en": "latin",
        "bn": "bengali",
        "gu": "gujarati",
        "pa": "gurmukhi",
        "or": "odia",
    }

    if lang == "sa":
        # Preserve homorganic nasal+stop clusters (ಙ್, ಞ್, ಣ್, ನ್, ಮ್) for accurate Sanskrit articulation
        native = _devanagari_to_target_script(phonemic_dev, "sa")
    elif lang == "en":
        if detected_script in ("latin", "iast"):
            native = s_raw.lower()
        else:
            native = _devanagari_to_target_script(phonemic_dev, "en")
    elif detected_script == script_for_lang.get(lang):
        native = s_raw
        if lang == "hi":
            native = delete_hindi_schwa(native)
        elif lang == "ml":
            for zwj_seq, atomic in MALAYALAM_ZWJ_CHILLU:
                native = native.replace(zwj_seq, atomic)
            native = (
                native.replace("\u200c", "")
                .replace("\u200d", "")
                .replace("\u0d4c", "\u0d57")
                .replace("\u0d7f", "\u0d15\u0d4d")
                .replace("\u0d0b", "\u0d31\u0d41")
            )
        elif lang == "te":
            native = (
                native.replace("\u0c0b", "\u0c30\u0c41")
                .replace("\u0c19", "\u0c02")
            )
        elif lang == "ta":
            native = native.lower().replace("\u0b83", "\u0bb9\u0bcd")
        elif lang == "pa":
            native = native.replace("\u0a74", "\u0a07\u0a71\u0a15 \u0a13\u0a05\u0a70\u0a15\u0a3e\u0a30")
    else:
        if any(ch.isdigit() for ch in s_raw) and not any(ch.isalpha() for ch in s_raw):
            native = s_raw
        else:
            native = _devanagari_to_target_script(phonemic_dev, lang)
        if lang == "hi":
            native = delete_hindi_schwa(native)
        elif lang == "ml":
            native = (
                native.replace("\u0d4c", "\u0d57")
                .replace("\u0d0b", "\u0d31\u0d41")
            )
        elif lang == "te":
            native = (
                native.replace("\u0c0b", "\u0c30\u0c41")
                .replace("\u0c19", "\u0c02")
            )

    # Normalize digits across Indic scripts: Punjabi uses native Gurmukhi digits in vocab; others map to ASCII
    norm_digits: list[str] = []
    for ch in native:
        cp = ord(ch)
        mapped_d = None
        for lo, hi, _ in BRAHMIC_BASES:
            if lo + 0x66 <= cp <= lo + 0x6F:
                if lang == "pa":
                    mapped_d = chr(0x0A66 + (cp - (lo + 0x66)))
                else:
                    mapped_d = str(cp - (lo + 0x66))
                break
        if mapped_d is None and lang == "pa" and "0" <= ch <= "9":
            mapped_d = chr(0x0A66 + int(ch))
        norm_digits.append(mapped_d if mapped_d is not None else ch)
    native = "".join(norm_digits)

    # All languages map sentence/clause punctuation to prosodic space boundaries
    native = re.sub(r"[\u0964\u0965|.,;:!?]+", " ", native)
    native = re.sub(r"\s+", " ", native).strip()
    return native


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
            ak_svara = None
            while i < n and (dev_text[i] in (ANUSVARA, VISARGA, CHANDRABINDU) or dev_text[i] in VEDIC_SVARA_MARKS):
                if dev_text[i] in VEDIC_SVARA_MARKS:
                    ak_svara = VEDIC_SVARA_MARKS[dev_text[i]]
                else:
                    if dev_text[i] in (ANUSVARA, VISARGA):
                        is_long = True
                    buf += dev_text[i]
                i += 1
            aksharas.append({"akshara": buf, "weight": "G" if is_long else "L", "svara": ak_svara})
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
                aksharas.append({"akshara": buf, "weight": "C", "svara": None})
                continue
            if i < n and dev_text[i] in DEV_MATRAS:
                if dev_text[i] in LONG_MATRAS:
                    is_long = True
                buf += dev_text[i]
                i += 1
            ak_svara = None
            while i < n and (dev_text[i] in (ANUSVARA, VISARGA, CHANDRABINDU) or dev_text[i] in VEDIC_SVARA_MARKS):
                if dev_text[i] in VEDIC_SVARA_MARKS:
                    ak_svara = VEDIC_SVARA_MARKS[dev_text[i]]
                else:
                    if dev_text[i] in (ANUSVARA, VISARGA):
                        is_long = True
                    buf += dev_text[i]
                i += 1
            j = i
            while j < n and dev_text[j].isspace():
                j += 1
            if j + 1 < n and dev_text[j] in DEV_CONSONANTS and dev_text[j + 1] == VIRAMA:
                is_long = True
            aksharas.append({"akshara": buf, "weight": "G" if is_long else "L", "svara": ak_svara})
            continue
        i += 1
    return aksharas


def text_to_ids(
    text: str,
    lang: str | None = None,
    add_blank: bool = True,
    min_len: int = 8,
    blank_id: int | None = None,
    legacy_blank: bool = False,
) -> dict[str, Any]:
    """Convert any Indic/Sanskrit/IAST/English string into blank-interleaved native specialist VITS input_ids + metadata."""
    detected_script, detected_lang = detect_script_and_lang(text, default_lang=lang or "sa")
    active_lang = lang if lang in LANGUAGE_IDS else detected_lang

    if active_lang == "en" and detected_script == "latin":
        dev = _english_to_devanagari(text)
    elif detected_script in ("iast", "latin"):
        dev = iast_to_devanagari(text)
    else:
        dev = brahmic_to_devanagari(text)

    if active_lang == "pa":
        dev = "".join(_PA_DIGIT_DEV.get(ch, ch) for ch in dev)
        dev = brahmic_to_devanagari(dev)

    aksharas = segment_aksharas(dev)
    phonemic = apply_phonological_rules(dev, lang=active_lang)
    tokens: list[str] = [ch for ch in phonemic if ch in TOKEN_TO_ID]
    if not tokens:
        tokens = ["\u0905"]

    native_text = to_native_model_text(text, active_lang, detected_script, phonemic)
    lang_map = NATIVE_TOKEN_TO_ID[active_lang]
    donor_map = NATIVE_DONOR_MAP.get(active_lang, {})
    raw_ids: list[int] = []
    native_tokens: list[str] = []
    for ch in native_text:
        if ch in lang_map:
            raw_ids.append(lang_map[ch])
            native_tokens.append(ch)
        elif ch in donor_map and donor_map[ch] in lang_map:
            raw_ids.append(lang_map[donor_map[ch]])
            native_tokens.append(ch)

    if not raw_ids:
        fallback_tok = {
            "sa": "\u0c85",
            "kn": "\u0c85",
            "hi": "\u0905",
            "mr": "\u0905",
            "ta": "\u0b85",
            "te": "\u0c05",
            "ml": "\u0d05",
            "en": "a",
            "bn": "\u0985",
            "gu": "\u0a85",
            "pa": "\u0a05",
            "or": "\u0b05",
        }[active_lang]
        raw_ids = [lang_map[fallback_tok]]
        native_tokens = [fallback_tok]

    effective_blank_id = (
        0 if legacy_blank else (blank_id if blank_id is not None else PAD_TOKEN_ID.get(active_lang, BLANK_ID))
    )

    if add_blank:
        sp_id = lang_map.get(" ")
        if sp_id is not None and raw_ids[-1] != sp_id:
            raw_ids.append(sp_id)
            native_tokens.append(" ")
        interleaved: list[int] = [effective_blank_id]
        is_blank_mask: list[bool] = [True]
        token_type_ids: list[int] = [1]
        for tid in raw_ids:
            interleaved.append(tid)
            is_blank_mask.append(False)
            token_type_ids.append(0)
            interleaved.append(effective_blank_id)
            is_blank_mask.append(True)
            token_type_ids.append(1)
        input_ids = interleaved
    else:
        input_ids = raw_ids
        is_blank_mask = [False] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

    while len(input_ids) < min_len:
        input_ids.append(effective_blank_id)
        is_blank_mask.append(True)
        token_type_ids.append(1)

    virama_tok = {
        "sa": "\u0ccd",
        "kn": "\u0ccd",
        "hi": "\u094d",
        "mr": "\u094d",
        "ta": "\u0bcd",
        "te": "\u0c4d",
        "ml": "\u0d4d",
        "bn": "\u09cd",
        "gu": "\u0acd",
        "pa": "\u0a4d",
        "or": "\u0b4d",
    }.get(active_lang, "")
    virama_id = lang_map.get(virama_tok, -1) if virama_tok else -1
    is_virama = [1 if tid == virama_id else 0 for tid in input_ids]

    ipa_str = text_to_ipa(text, lang=active_lang, normalized_dev_fallback=dev if active_lang != "en" else None)
    ipa_ids = ipa_to_ids(ipa_str)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "ipa": ipa_str,
        "ipa_ids": ipa_ids,
        "is_blank": is_blank_mask,
        "token_type_ids": token_type_ids,
        "is_virama": is_virama,
        "svaras": [a.get("svara") for a in aksharas],
        "lang": active_lang,
        "lang_id": LANGUAGE_IDS[active_lang],
        "script": detected_script,
        "normalized_devanagari": phonemic,
        "native_text": native_text,
        "tokens": tokens,
        "native_tokens": native_tokens,
        "aksharas": aksharas,
    }
