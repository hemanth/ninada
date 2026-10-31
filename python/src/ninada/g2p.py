"""Pan-Indic Akshara, Per-Language Native Brahmic, and Classical/Vedic Sanskrit Phoneme G2P for Ninada."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

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
BLANK_ID: int = 0
UNK_ID: int = TOKEN_TO_ID["<unk>"]

LANGUAGE_IDS: dict[str, int] = {
    "sa": 0,  # Sanskrit (Classical & Vedic, zero-schwa-deletion Kannada-isomorphic phoneme map)
    "hi": 1,  # Hindi (Native Devanagari)
    "kn": 2,  # Kannada (Native Kannada)
    "ta": 3,  # Tamil (Native Tamil)
    "te": 4,  # Telugu (Native Telugu)
    "ml": 5,  # Malayalam (Native Malayalam with atomic chillu letters)
}
ID_TO_LANGUAGE: dict[int, str] = {v: k for k, v in LANGUAGE_IDS.items()}

# Exact per-language native specialist vocabularies (blank_id == 0 across all 6 languages)
NATIVE_VOCABS: dict[str, list[str]] = {
    "kn": [
        "'", "ಓ", "ಮ", "ಆ", "ಭ", "ಋ", "ಚ", "ಟ", "ು", "ೇ",
        "ಜ", "ಏ", "ಾ", "ಪ", "ಶ", "್", "ಈ", "ಔ", "ಞ", "1",
        "ನ", "ಎ", "ಸ", "ವ", "ಇ", "ಘ", "5", "ಊ", "ಳ", "ಖ",
        "ೃ", "ೆ", "-", "ತ", "ಝ", "ಢ", " ", "ಅ", "ಧ", "ೌ",
        "ಹ", "ಯ", "9", "ಫ", "ಛ", "8", "6", "ಲ", "ಣ", "_",
        "ದ", "ಬ", "ಠ", "ಡ", "ಃ", "3", "ೋ", "ಂ", "ಒ", "ೀ",
        "ಕ", "ಿ", "ಥ", "ರ", "ೂ", "ಐ", "ೈ", "ಉ", "2", "0",
        "ಗ", "ೊ", "ಷ", "4", "7", "<unk>",
    ],
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
    "te": [
        "త", "ఞ", "భ", "థ", "ర", "న", "ో", "ఘ", "'", "ఐ",
        "గ", "ే", "జ", "ఔ", "హ", "ా", "డ", "మ", "స", " ",
        "క", "ఒ", "ూ", "వ", "ధ", "శ", "ౌ", "ప", "ః", "ఠ",
        "ఖ", "ష", "–", "ణ", "ఫ", "ి", "ొ", "ఏ", "ఎ", "ృ",
        "ఓ", "య", "అ", "ీ", "ె", "_", "చ", "ు", "్", "ద",
        "ల", "ఉ", "ఢ", "-", "బ", "ఈ", "ట", "ఛ", "ళ", "ఊ",
        "6", "ం", "ఇ", "ఆ", "ై", "<unk>",
    ],
    "ml": [
        "—", "ത", "ര", "ള", "ഴ", "ഒ", "0", "ൂ", "ഖ", "ഞ",
        "ഇ", "ഡ", "ി", "ഊ", "ച", "ാ", "യ", "ഷ", "i", "ഉ",
        "ഫ", "ല", "ഠ", "4", "െ", "ൗ", "ആ", "സ", "ു", "ം",
        "m", "ജ", "ധ", "-", "ഹ", "ഛ", "്", "ട", "ഃ", "ഗ",
        "ണ", "ൽ", "ോ", "_", "റ", "ഏ", "6", "എ", "ർ", "ഥ",
        "ഘ", "ഓ", "ീ", "ൈ", "2", "േ", "ക", "1", "ൾ", "ങ",
        "ഭ", " ", "ഢ", "അ", "മ", "ബ", "3", "ഈ", "ഔ", "പ",
        "ൺ", "ന", "ൊ", "o", "ൃ", "c", "വ", "ൻ", "ദ", "5",
        "ഐ", "'", "q", "ശ", "<unk>",
    ],
}
# Sanskrit uses the full zero-schwa-deletion Kannada phoneme inventory (0..75) + 3 Vedic/Classical extensions (76..78)
NATIVE_VOCABS["sa"] = NATIVE_VOCABS["kn"] + ["\u0c99", "\u0ce0", "\u0cc4"]

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
}

# Articulatory donor characters for Sanskrit's 3 extended tokens (76..78) on facebook/mms-tts-kan
SANSKRIT_DONOR_MAP: dict[str, str] = {
    "\u0c99": "\u0ca3",  # ಙ (ṅa) <- ಣ (ṇa)
    "\u0ce0": "\u0c8b",  # ೠ (ṝ)  <- ಋ (ṛ)
    "\u0cc4": "\u0cc3",  # ೄ (ṝ)  <- ೃ (ṛ)
}

# Kept for backward compatibility
WARMSTART_DONOR_MAP: dict[str, str] = dict(SANSKRIT_DONOR_MAP)

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
    for zwj_seq, atomic in MALAYALAM_ZWJ_CHILLU:
        norm = norm.replace(zwj_seq, atomic)
    norm = norm.replace("\u0d4c", "\u0d57")
    out: list[str] = []
    for ch in norm:
        cp = ord(ch)
        if cp in MALAYALAM_CHILLU_CP:
            out.append(MALAYALAM_CHILLU_CP[cp])
            continue
        if cp == 0x0D57:
            out.append("\u094c")
            continue
        if 0x0900 <= cp <= 0x097F:
            out.append(ch)
            continue
        mapped = False
        for lo, hi, _ in BRAHMIC_BASES[1:]:
            if lo <= cp <= hi:
                offset = cp - lo
                if 0x01 <= offset <= 0x4D or offset in (0x60, 0x62, 0x63):
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
    s = unicodedata.normalize("NFC", dev_text)
    s = re.sub(r"[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]", "", s)
    s = s.replace("\u1cf5", VISARGA).replace("\u1cf6", VISARGA)
    s = s.replace("\u0950", "\u0913\u092e\u094d")
    s = re.sub(r"([\u0947\u094b\u093e\u090f\u0913\u0906])\u093d+", r"\1", s)
    s = s.replace("\u093d", "\u0905")
    s = s.replace("\u0962", "\u0943").replace("\u0963", "\u0944").replace("\u090c", "\u0932\u0943")

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


def _devanagari_to_target_script(dev_text: str, target_lang: str) -> str:
    """Transliterate canonical Devanagari into target native Brahmic script when input was IAST or cross-script."""
    if target_lang == "hi":
        return dev_text.replace("\u0964", " - ").replace("\u0965", " - ")

    base_map = {
        "sa": 0x0C80,  # Kannada-isomorphic zero-schwa-deletion Sanskrit
        "kn": 0x0C80,
        "ta": 0x0B80,
        "te": 0x0C00,
        "ml": 0x0D00,
    }
    target_base = base_map.get(target_lang, 0x0C80)
    s = dev_text.replace("\u0901", "\u0902")

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
            out.append(" " if target_lang == "ta" else " - ")
            continue
        cp = ord(ch)
        if cp == 0x093C:
            continue  # strip Devanagari nukta in Southern Brahmic targets
        if 0x0901 <= cp <= 0x0963:
            offset = cp - 0x0900
            # Short e/o fallback: if Devanagari had long ए/ओ (0x0F/0x13, 0x47/0x4B), keep long ಏ/ಓ in sa/kn/te/ml/ta
            out.append(chr(target_base + offset))
        else:
            out.append(ch)
    return "".join(out)


def to_native_model_text(text: str, lang: str, detected_script: str, phonemic_dev: str) -> str:
    """Produce normalized native-script text matching the target specialist model's exact vocabulary."""
    s_raw = unicodedata.normalize("NFC", text)
    script_for_lang = {
        "kn": "kannada",
        "hi": "devanagari",
        "ta": "tamil",
        "te": "telugu",
        "ml": "malayalam",
    }

    if lang == "sa":
        native = _devanagari_to_target_script(phonemic_dev, "sa")
        # Kannada VITS orthography encodes homorganic nasal+stop clusters via Anusvara (ಂ)
        native = re.sub(
            r"[\u0c99\u0c9e\u0ca3\u0ca8\u0cae]\u0ccd(?=[\u0c95-\u0c98\u0c9a-\u0c9d\u0c9f-\u0ca2\u0ca4-\u0ca7\u0caa-\u0cad])",
            "\u0c82",
            native,
        )
    elif detected_script == script_for_lang.get(lang):
        native = s_raw
        if lang == "ml":
            for zwj_seq, atomic in MALAYALAM_ZWJ_CHILLU:
                native = native.replace(zwj_seq, atomic)
            native = (
                native.replace("\u200c", "")
                .replace("\u200d", "")
                .replace("\u0d4c", "\u0d57")
                .replace("\u0d7f", "\u0d15\u0d4d")
                .replace("\u0d0b", "\u0d31\u0d41")
                .replace("\u0d1d", "\u0d1c")
            )
        elif lang == "te":
            native = (
                native.replace("\u0c0b", "\u0c30\u0c41")
                .replace("\u0c1d", "\u0c1c")
                .replace("\u0c19", "\u0c02")
            )
        elif lang == "ta":
            native = native.lower().replace("\u0b83", "\u0bb9\u0bcd")
        elif lang == "kn":
            native = native.replace("\u0c99", "\u0ca8")
    else:
        native = _devanagari_to_target_script(phonemic_dev, lang)
        if lang == "ml":
            native = (
                native.replace("\u0d4c", "\u0d57")
                .replace("\u0d0b", "\u0d31\u0d41")
                .replace("\u0d1d", "\u0d1c")
            )
        elif lang == "te":
            native = (
                native.replace("\u0c0b", "\u0c30\u0c41")
                .replace("\u0c1d", "\u0c1c")
                .replace("\u0c19", "\u0c02")
            )
        elif lang == "kn":
            native = native.replace("\u0c99", "\u0ca8")

    # Normalize digits across Indic scripts to ASCII digits where present in vocab
    norm_digits: list[str] = []
    for ch in native:
        cp = ord(ch)
        mapped_d = None
        for lo, hi, _ in BRAHMIC_BASES:
            if lo + 0x66 <= cp <= lo + 0x6F:
                mapped_d = str(cp - (lo + 0x66))
                break
        norm_digits.append(mapped_d if mapped_d is not None else ch)
    native = "".join(norm_digits)

    sep = " " if lang == "ta" else " - "
    native = re.sub(r"[\u0964\u0965|.,;:!?]+", sep, native)
    native = re.sub(r"\s*-\s*(?:-\s*)*", " - ", native) if lang != "ta" else native
    native = re.sub(r"\s+", " ", native).strip(" -")
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
    """Convert any Indic/Sanskrit/IAST string into blank-interleaved native specialist VITS input_ids + metadata."""
    detected_script, detected_lang = detect_script_and_lang(text, default_lang=lang or "sa")
    active_lang = lang if lang in LANGUAGE_IDS else detected_lang

    if detected_script in ("iast", "latin"):
        dev = iast_to_devanagari(text)
    else:
        dev = brahmic_to_devanagari(text)

    phonemic = apply_phonological_rules(dev, lang=active_lang)
    tokens: list[str] = [ch for ch in phonemic if ch in TOKEN_TO_ID]
    if not tokens:
        tokens = ["\u0905"]

    native_text = to_native_model_text(text, active_lang, detected_script, phonemic)
    lang_map = NATIVE_TOKEN_TO_ID[active_lang]
    raw_ids: list[int] = []
    native_tokens: list[str] = []
    for ch in native_text:
        if ch in lang_map:
            raw_ids.append(lang_map[ch])
            native_tokens.append(ch)

    if not raw_ids:
        fallback_tok = {
            "sa": "\u0c85",
            "kn": "\u0c85",
            "hi": "\u0905",
            "ta": "\u0b85",
            "te": "\u0c05",
            "ml": "\u0d05",
        }[active_lang]
        raw_ids = [lang_map[fallback_tok]]
        native_tokens = [fallback_tok]

    if add_blank:
        sp_id = lang_map.get(" ")
        if sp_id is not None and raw_ids[-1] != sp_id:
            raw_ids.append(sp_id)
            native_tokens.append(" ")
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
        "native_text": native_text,
        "tokens": tokens,
        "native_tokens": native_tokens,
        "aksharas": segment_aksharas(phonemic),
    }
