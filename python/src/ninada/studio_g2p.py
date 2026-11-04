"""
Ninada v5.0 — Native Per-Language 24,000 Hz Studio VITS Character Tokenizer.

Maps native Unicode orthography directly to each specialist's exact character vocabulary
(`sa`: 144, `hi`: 120, `kn`: 85, `ta`: 61, `te`: 110, `ml`: 101, `en`: 102, `mr`: 144,
`bn`: 100, `gu`: 124, `pa`: 85, `or`: 94 tokens) with zero Hindi pivot, zero lossy
transliteration when input is in native script, and automatic IAST / cross-script fallback
when Roman or cross-script text is supplied.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .g2p import (
    delete_hindi_schwa,
    delete_hindi_schwa_word,
    detect_script_and_lang,
    iast_to_devanagari,
    to_native_model_text,
)

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
    "en": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": 'â,pGYI&“US7(9…Z–v1yadHND`@lw”80ti?42mOgCE_XRMz\ufe0fk3‘VeP5jè"rK#F’)scQ\'Jôfq xA»bTouhnBWL6àé',
        "blank": "<BLNK>",
    },
    "mr": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": ')ऊभॐब८•औॲ?z५ी/ऩठअवइ\',डळत3° ञोnआफ़ॊच:․छऍई;ओuगझल0२ॆऑऱ!ॕ"(मर|9…न-ॄु5ृहेॅःएघ७शऎ1ङॠं।१ष४cॉधाऽ\u200aकि2.ऋउ–ूa्स६जखदथ०68णऐैफ़ँs९३4ौ7—टपढयऌॡॢॣ॥',
        "blank": "<BLNK>",
    },
    "bn": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": 'এাসৎঢঊচঢ়রখৰঘড়ঋ"ঔহনবভঅঠ্ঃৌছৃিড়কতীেওলং, জউ?থঞ\'ফুআঐয়ইূৈযগঁঙঝ।ঈৗণষধশটপদোমঌঽৄৠৡ০১২৩৪৫৬৭৮৯',
        "blank": "<BLNK>",
    },
    "gu": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": 'ધઋલ૭ઉઊઝઈછઍત6ગेફ“78ખથरયજન૧ળણ,ૉ}ઞ૪ઔશચ3ા૦"(ૅ્મષૠ૩સ‘अઇઠુक૫દોર\'ીપૈ૬઼मઆ”ૌ4)ૂિહ{ઑઓ કઢઃ2ૃેવભि?અએ5૯ઘટંડઐ1૮ઁબ’09૨ઌઙઽૄૐૡ',
        "blank": "<BLNK>",
    },
    "pa": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": "ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੜੰੱੲੳ੦੧੨੩੪੫੬੭੮੯",
        "blank": "<BLNK>",
    },
    "or": {
        "pad": "<PAD>",
        "punctuations": "!¡'(),-.:;¿? ",
        "characters": "ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହ଼ଽାିୀୁୂୃୄେୈୋୌ୍ୖୗୟୠୡୱ୦୧୨୩୪୫୬୭୮୯",
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
    "mr": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "or": (0x0B00, 0x0B7F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "en": (0x0041, 0x007A),
}
_EN_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_ORD_MAP = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_EN_ABBREVS = (
    (re.compile(r"\bmrs\.", re.IGNORECASE), "missus"),
    (re.compile(r"\bmr\.", re.IGNORECASE), "mister"),
    (re.compile(r"\bms\.", re.IGNORECASE), "miss"),
    (re.compile(r"\bdr\.", re.IGNORECASE), "doctor"),
    (re.compile(r"\bprof\.", re.IGNORECASE), "professor"),
    (re.compile(r"\bst\.", re.IGNORECASE), "saint"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\binc\.", re.IGNORECASE), "incorporated"),
    (re.compile(r"\bcorp\.", re.IGNORECASE), "corporation"),
    (re.compile(r"\bltd\.", re.IGNORECASE), "limited"),
    (re.compile(r"\bco\.", re.IGNORECASE), "company"),
    (re.compile(r"\bjr\.", re.IGNORECASE), "junior"),
    (re.compile(r"\bsr\.", re.IGNORECASE), "senior"),
    (re.compile(r"\bdept\.", re.IGNORECASE), "department"),
    (re.compile(r"\buniv\.", re.IGNORECASE), "university"),
    (re.compile(r"\bno\.", re.IGNORECASE), "number"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\brs\.", re.IGNORECASE), "rupees"),
)


def _en_int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _en_int_to_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _EN_TENS[t] if r == 0 else f"{_EN_TENS[t]} {_EN_ONES[r]}"
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_EN_ONES[h]} hundred" if r == 0 else f"{_EN_ONES[h]} hundred {_en_int_to_words(r)}"
    if 1900 <= n <= 2099 and n % 100 != 0 and not (2001 <= n <= 2009):
        hi, lo = divmod(n, 100)
        lo_str = f"oh {_EN_ONES[lo]}" if lo < 10 else _en_int_to_words(lo)
        return f"{_en_int_to_words(hi)} {lo_str}"
    if n < 1_000_000:
        k, r = divmod(n, 1000)
        return f"{_en_int_to_words(k)} thousand" if r == 0 else f"{_en_int_to_words(k)} thousand {_en_int_to_words(r)}"
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000)
        return f"{_en_int_to_words(m)} million" if r == 0 else f"{_en_int_to_words(m)} million {_en_int_to_words(r)}"
    if n < 1_000_000_000_000:
        b, r = divmod(n, 1_000_000_000)
        return f"{_en_int_to_words(b)} billion" if r == 0 else f"{_en_int_to_words(b)} billion {_en_int_to_words(r)}"
    return " ".join(_EN_ONES[int(d)] for d in str(n))


def _en_ordinal_to_words(n: int) -> str:
    words = _en_int_to_words(n).split()
    last = words[-1]
    if last in _EN_ORD_MAP:
        words[-1] = _EN_ORD_MAP[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _normalize_english_text(t: str) -> str:
    t = re.sub(r"(?<=\d)\s*kHz\b", " kilohertz", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*MHz\b", " megahertz", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*GHz\b", " gigahertz", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkHz\b", "kilohertz", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMHz\b", "megahertz", t, flags=re.IGNORECASE)
    t = re.sub(r"\bGHz\b", "gigahertz", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*Hz\b", " hertz", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*ms\b", " milliseconds", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*MB\b", " megabytes", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*GB\b", " gigabytes", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d)\s*KB\b", " kilobytes", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b(\d{1,12})(st|nd|rd|th)\b",
        lambda m: _en_ordinal_to_words(int(m.group(1))),
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([A-Za-z])", r"\1 \2", t)
    for pat, rep in _EN_ABBREVS:
        t = pat.sub(rep, t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)

    def _expand_curr_dec(m: re.Match[str]) -> str:
        sym = m.group(1)
        dollars = int(m.group(2))
        cents = int(m.group(3))
        curr_sing, curr_plur, sub_sing, sub_plur = {
            "$": ("dollar", "dollars", "cent", "cents"),
            "€": ("euro", "euros", "cent", "cents"),
            "₹": ("rupee", "rupees", "paise", "paise"),
        }[sym]
        parts = []
        if dollars > 0 or cents == 0:
            c_name = curr_sing if dollars == 1 else curr_plur
            parts.append(f"{_en_int_to_words(dollars)} {c_name}")
        if cents > 0:
            s_name = sub_sing if cents == 1 else sub_plur
            if parts:
                parts.append(f"and {_en_int_to_words(cents)} {s_name}")
            else:
                parts.append(f"{_en_int_to_words(cents)} {s_name}")
        return " ".join(parts)

    def _expand_curr_int(m: re.Match[str]) -> str:
        sym = m.group(1)
        val = int(m.group(2))
        curr_sing, curr_plur = {
            "$": ("dollar", "dollars"),
            "€": ("euro", "euros"),
            "₹": ("rupee", "rupees"),
        }[sym]
        c_name = curr_sing if val == 1 else curr_plur
        return f"{_en_int_to_words(val)} {c_name}"

    t = re.sub(r"([\$€₹])(\d+)\.(\d{2})\b", _expand_curr_dec, t)
    t = re.sub(r"([\$€₹])(\d+)\b", _expand_curr_int, t)
    t = t.replace("&", " and ").replace("%", " percent ").replace("₹", " rupees ").replace("$", " dollars ").replace("€", " euros ")

    def _expand_dotted(m: re.Match[str]) -> str:
        s = m.group(0)
        parts = s.split(".")
        if len(parts) == 2:
            whole = _en_int_to_words(int(parts[0]))
            frac_words = " ".join(_EN_ONES[int(d)] for d in parts[1])
            return f"{whole} point {frac_words}"
        out_parts = []
        for p in parts:
            if p == "0":
                out_parts.append("zero")
            elif len(p) > 1 and p.startswith("0"):
                out_parts.append(" ".join(_EN_ONES[int(d)] for d in p))
            else:
                out_parts.append(_en_int_to_words(int(p)))
        return " point ".join(out_parts)

    t = re.sub(r"\b\d+(?:\.\d+)+\b", _expand_dotted, t)
    t = re.sub(r"\b(\d{1,12})\b", lambda m: _en_int_to_words(int(m.group(1))), t)
    return t


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
    """Normalize native Indic/Sanskrit/English text identically to `StudioCharTokenizer`."""
    lc = lang if lang in STUDIO_VOCABS else detect_script_and_lang(text, default_lang="sa")[1]
    t = unicodedata.normalize("NFC", str(text or "")).strip()
    if not t:
        return ""

    if lc == "en":
        t = _normalize_english_text(t)
    else:
        lo, hi = _SCRIPT_RANGES[lc]
        has_target_script = any(lo <= ord(ch) <= hi for ch in t)
        if not has_target_script:
            if lc == "sa":
                t = iast_to_devanagari(t)
            else:
                det_sc = detect_script_and_lang(t, default_lang=lc)[0]
                dev = iast_to_devanagari(t) if det_sc in ("iast", "latin") else t
                t = to_native_model_text(t, lang=lc, detected_script=det_sc, phonemic_dev=dev)

    if lc == "sa":
        t = re.sub(r"[\u0951-\u0954\u1cd0-\u1cff]", "", t)
        t = re.sub(r"\s*ऽ+", "ऽ", t)
        t = t.replace("||", "॥").replace("|", "।")

    if lc == "hi":
        t = re.sub(r"[\u0951-\u0954\u1cd0-\u1cff]", "", t)
        t = t.replace("||", "।").replace("|", "।").replace("।।", "।")

    if lc == "mr":
        t = re.sub(r"[\u0951-\u0954\u1cd0-\u1cff]", "", t)
        t = t.replace("॥", "।")

    if lc == "bn":
        t = t.replace("\u09f1", "ব").replace("॥", "।")

    if lc == "gu":
        t = t.replace("॥", ".").replace("।", ".")

    if lc == "pa":
        t = t.replace("\u0a74", "ਇੱਕ ਓਅੰਕਾਰ")
        t = t.replace("॥", ".").replace("।", ".")

    if lc == "or":
        t = t.replace("॥", ".").replace("।", ".")

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
    if lc in ("kn", "ta", "te", "ml", "en", "gu", "pa", "or"):
        t = t.replace("॥", ".").replace("।", ".")
    elif lc in ("hi", "mr", "bn"):
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


__all__ = [
    "STUDIO_HOP_LENGTH",
    "STUDIO_SAMPLE_RATE",
    "STUDIO_SPECS",
    "STUDIO_VOCABS",
    "clean_studio_text",
    "studio_text_to_ids",
    "delete_hindi_schwa",
    "delete_hindi_schwa_word",
]

