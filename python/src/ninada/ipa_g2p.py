"""Zero-dependency Pan-Indic & Classical/Vedic Sanskrit -> 24 kHz IPA Phonemizer for Ninada v3.0.

Implements deterministic phonological rules across all six supported scripts/languages:
  - Sanskrit (`sa`): Full inherent-schwa preservation, Paninian homorganic anusvara, contextual visarga,
    avagraha vowel prolongation, Vedic svara accent stress, and true diphthongs (`aɪ`, `aʊ`).
  - Hindi (`hi`): Ohala (1983) / Choudhury medial & word-final schwa deletion (with cluster-release guard),
    nukta consonants, chandrabindu nasalization, and standard monophthongal `æː` / `ɔː`.
  - Kannada (`kn`): Full word-final vowel retention, 5-way short/long `e/eː` & `o/oː`, retroflex lateral `ɭ`,
    and homorganic sunna assimilation.
  - Tamil (`ta`): Full Caldwell's Law positional allophony (geminate voiceless stops, post-nasal stop voicing,
    intervocalic/post-liquid lenition, word-initial `ச -> s`), Grantha consonants, `ழ -> ɻ`, and `ள -> ɭ`.
  - Telugu (`te`): Full *ajanta* word-final vowel retention, 5-way short/long `e/eː` & `o/oː`, `ళ -> ɭ`,
    aspirated stops, and homorganic sunna assimilation.
  - Malayalam (`ml`): Atomic chillaksharam (`ൺ, ൻ, ർ, ൽ, ൾ, ൿ`), word-final *samvruthokaram* (`് -> ᵊ`),
    post-nasal stop voicing (`ങ്ക, ഞ്ച, ണ്ട, ന്ത, മ്പ, ന്റ`), `റ്റ -> tt`, `ഴ -> ɻ`, `ള -> ɭ`, and `റ=r` vs `ര=ɾ`.
"""
from __future__ import annotations

import re
import unicodedata

# Exact 178-slot StyleTTS2 / PL-BERT / iSTFTNet 24 kHz IPA Vocabulary
# Slot 79 ('ɭ') is initialized from lateral+retroflex centroid and fine-tuned on GCP studio corpora.
IPA_VOCAB: dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10,
    '"': 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "̃": 17,
    "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23,
    "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41,
    "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50,
    "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58,
    "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66,
    "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76,
    "ɕ": 77, "ç": 78, "ɭ": 79, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85,
    "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103,
    "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116,
    "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128,
    "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136,
    "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148,
    "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164,
    "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177,
}
IPA_ID_TO_TOKEN: dict[int, str] = {v: k for k, v in IPA_VOCAB.items()}
RETROFLEX_LATERAL_ID: int = 79

# Canonical Devanagari codepoints (all 6 scripts map losslessly into extended Devanagari offsets 0x00..0x7F)
_VIRAMA = "\u094d"
_NUKTA = "\u093c"
_ANUSVARA = "\u0902"
_CHANDRABINDU = "\u0901"
_VISARGA = "\u0903"
_AVAGRAHA = "\u093d"

_NUKTA_PRECOMPOSED: dict[str, str] = {
    "\u0958": "q",    # क़
    "\u0959": "x",    # ख़
    "\u095a": "ɣ",    # ग़
    "\u095b": "z",    # ज़
    "\u095c": "ɽ",    # ड़
    "\u095d": "ɽʰ",   # ढ़
    "\u095e": "f",    # फ़
    "\u095f": "j",    # य़
}

_NUKTA_BASE_MAP: dict[str, str] = {
    "क": "q",
    "ख": "x",
    "ग": "ɣ",
    "ज": "z",
    "ड": "ɽ",
    "ढ": "ɽʰ",
    "फ": "f",
}

_CONSONANT_IPA: dict[str, str] = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    "च": "ʧ", "छ": "ʧʰ", "ज": "ʤ", "झ": "ʤʰ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    "त": "t", "थ": "tʰ", "द": "d", "ध": "dʰ", "न": "n", "\u0929": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    "य": "j", "र": "ɾ", "\u0931": "r", "ल": "l", "ळ": "ɭ", "\u0934": "ɻ",
    "व": "ʋ", "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "h",
}

_STOP_HOMORGANIC_NASAL: dict[str, str] = {
    "क": "ŋ", "ख": "ŋ", "ग": "ŋ", "घ": "ŋ", "ङ": "ŋ",
    "च": "ɲ", "छ": "ɲ", "ज": "ɲ", "झ": "ɲ", "ञ": "ɲ",
    "ट": "ɳ", "ठ": "ɳ", "ड": "ɳ", "ढ": "ɳ", "ण": "ɳ",
    "त": "n", "थ": "n", "द": "n", "ध": "n", "न": "n", "\u0929": "n",
    "प": "m", "फ": "m", "ब": "m", "भ": "m", "म": "m",
}

# Malayalam Chillu -> pure coda consonant (no inherent vowel)
_ML_CHILLU_IPA: dict[str, str] = {
    "\u0d7a": "ɳ",  # ൺ
    "\u0d7b": "n",  # ൻ
    "\u0d7c": "ɾ",  # ർ
    "\u0d7d": "l",  # ൽ
    "\u0d7e": "ɭ",  # ൾ
    "\u0d7f": "k",  # ൿ
}

_INDEP_VOWELS_COMMON: dict[str, str] = {
    "अ": "ə",
    "आ": "ɑː",
    "इ": "ɪ",
    "ई": "iː",
    "उ": "ʊ",
    "ऊ": "uː",
    "ऋ": "ɾɪ",
    "ॠ": "ɾiː",
    "ऌ": "lɪ",
    "ॡ": "liː",
    "\u090e": "ɛ",   # short e (Dravidian എ/ఎ/எ/ಎ -> espeak-native [ɛ])
    "ए": "eː",
    "ऍ": "æː",
    "\u0912": "o",   # short o (Dravidian ഒ/ఒ/ஒ/ഒ)
    "ओ": "oː",
    "ऑ": "ɔː",
}

_MATRAS_COMMON: dict[str, str] = {
    "ा": "ɑː",
    "ि": "ɪ",
    "ी": "iː",
    "ु": "ʊ",
    "ू": "uː",
    "ृ": "ɾɪ",
    "ॄ": "ɾiː",
    "ॢ": "lɪ",
    "ॣ": "liː",
    "\u0946": "ɛ",   # short e matra (Dravidian െ/ె/ெ/ೆ -> espeak-native [ɛ])
    "े": "eː",
    "ॅ": "æː",
    "\u094a": "o",   # short o matra
    "ो": "oː",
    "ॉ": "ɔː",
}

_LONG_VOWEL_IPA = {"ɑː", "iː", "uː", "ɾiː", "eː", "ɛː", "æː", "oː", "ɔː", "aɪ", "əɪ", "aʊ", "əʊ"}
_VOWEL_CHARS = set("əɐʌɑɪiuʊeɛæoɔaᵊ")
_RELEASE_TAIL_CHARS = _VOWEL_CHARS | {"ː", "\u0303"}


def _vowel_ipa(ch: str, is_matra: bool, lang: str) -> str:
    if is_matra:
        if ch == "ै":
            return "ɛː" if lang == "hi" else "əɪ"
        if ch == "ौ":
            return "ɔː" if lang == "hi" else "əʊ"
        return _MATRAS_COMMON.get(ch, "")
    if ch == "ऐ":
        return "ɛː" if lang == "hi" else "əɪ"
    if ch == "औ":
        return "ɔː" if lang == "hi" else "əʊ"
    return _INDEP_VOWELS_COMMON.get(ch, "")


def _normalize_script_word_to_dev(word: str, lang: str) -> str:
    """Losslessly project Kannada/Tamil/Telugu/Malayalam codepoints to extended Devanagari while keeping chillu."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        if ch in _ML_CHILLU_IPA:
            out.append(ch)
            continue
        cp = ord(ch)
        if 0x0B80 <= cp <= 0x0BFF:      # Tamil
            if ch == "\u0b83":          # ஃ āytam
                out.append("ख़")
            else:
                out.append(chr(cp - 0x0B80 + 0x0900))
        elif 0x0C00 <= cp <= 0x0C7F:    # Telugu
            out.append(chr(cp - 0x0C00 + 0x0900))
        elif 0x0C80 <= cp <= 0x0CFF:    # Kannada
            out.append(chr(cp - 0x0C80 + 0x0900))
        elif 0x0D00 <= cp <= 0x0D7F:    # Malayalam
            if ch == "\u0d3a":
                out.append("ट")
            else:
                out.append(chr(cp - 0x0D00 + 0x0900))
        else:
            out.append(ch)
    return "".join(out)


def _parse_word_units(dev_word: str, lang: str) -> list[dict[str, object]]:
    """Parse a single normalized Devanagari (+ chillu) word into structured phonetic units."""
    units: list[dict[str, object]] = []
    n = len(dev_word)
    i = 0
    while i < n:
        ch = dev_word[i]
        if ch in ("\u200c", "\u200d"):
            i += 1
            continue
        if ch in _ML_CHILLU_IPA:
            units.append({
                "kind": "C",
                "cons": _ML_CHILLU_IPA[ch],
                "raw_c": ch,
                "vowel": "",
                "inherent": False,
                "nasal": "",
                "visarga": False,
                "stress": False,
                "final_virama": True,
            })
            i += 1
            continue
        if ch in ("॑", "॓"):
            if units:
                units[-1]["stress"] = True
            i += 1
            continue
        if ch == "॒":
            i += 1
            continue
        if ch == _AVAGRAHA:
            if units and isinstance(units[-1].get("vowel"), str) and units[-1]["vowel"]:
                v = str(units[-1]["vowel"])
                if not v.endswith("ː"):
                    units[-1]["vowel"] = v + "ː"
            i += 1
            continue

        # Independent vowel
        v_ind = _vowel_ipa(ch, is_matra=False, lang=lang)
        if v_ind:
            i += 1
            nasal = ""
            visarga = False
            stress = False
            while i < n and dev_word[i] in (_ANUSVARA, _CHANDRABINDU, _VISARGA, "॑", "॒", "॓"):
                m = dev_word[i]
                if m == _CHANDRABINDU:
                    nasal = "̃"
                elif m == _ANUSVARA:
                    nasal = "M"
                elif m == _VISARGA:
                    visarga = True
                elif m in ("॑", "॓"):
                    stress = True
                i += 1
            units.append({
                "kind": "V",
                "cons": "",
                "raw_c": "",
                "vowel": v_ind,
                "inherent": False,
                "nasal": nasal,
                "visarga": visarga,
                "stress": stress,
                "final_virama": False,
            })
            continue

        # Consonant (with optional nukta)
        c_ipa = _NUKTA_PRECOMPOSED.get(ch)
        raw_c = ch
        if c_ipa is not None:
            i += 1
        elif ch in _CONSONANT_IPA:
            if i + 1 < n and dev_word[i + 1] == _NUKTA:
                c_ipa = _NUKTA_BASE_MAP.get(ch, _CONSONANT_IPA[ch])
                i += 2
            else:
                c_ipa = _CONSONANT_IPA[ch]
                if lang == "ta" and ch == "\u0931":
                    # Tamil single ற is alveolar tap/rhotic unless geminated ற்ற -> ttɾ
                    c_ipa = "ɾ"
                i += 1
        else:
            i += 1
            continue

        # Sanskrit ज् + ञ -> ʤɲ (handled naturally by C(ʤ, '') + C(ɲ, v)), क्ष -> k + ʂ (handled naturally!)
        if i < n and dev_word[i] == _VIRAMA:
            i += 1
            units.append({
                "kind": "C",
                "cons": c_ipa,
                "raw_c": raw_c,
                "vowel": "",
                "inherent": False,
                "nasal": "",
                "visarga": False,
                "stress": False,
                "final_virama": (i == n),
            })
            continue

        vowel = "ə"
        inherent = True
        if i < n:
            v_mat = _vowel_ipa(dev_word[i], is_matra=True, lang=lang)
            if v_mat:
                vowel = v_mat
                inherent = False
                i += 1

        nasal = ""
        visarga = False
        stress = False
        while i < n and dev_word[i] in (_ANUSVARA, _CHANDRABINDU, _VISARGA, "॑", "॒", "॓"):
            m = dev_word[i]
            if m == _CHANDRABINDU:
                nasal = "̃"
            elif m == _ANUSVARA:
                nasal = "M"
            elif m == _VISARGA:
                visarga = True
            elif m in ("॑", "॓"):
                stress = True
            i += 1

        units.append({
            "kind": "C",
            "cons": c_ipa,
            "raw_c": raw_c,
            "vowel": vowel,
            "inherent": inherent,
            "nasal": nasal,
            "visarga": visarga,
            "stress": stress,
            "final_virama": False,
        })

    return units


def _apply_language_allophony(units: list[dict[str, object]], lang: str) -> None:
    """Apply language-specific phonological rules in-place over parsed syllable/consonant units."""
    if not units:
        return

    # 1. Resolve ज्ञ / ಜ್ಞ / జ్ఞ (ज + ् + ञ -> ɡj in Hindi, ɡɲ in Sanskrit/Dravidian)
    for idx in range(len(units) - 1):
        u_cur = units[idx]
        u_nxt = units[idx + 1]
        if u_cur["raw_c"] == "ज" and u_cur["vowel"] == "" and u_nxt["raw_c"] == "ञ":
            u_cur["cons"] = "ɡ"
            u_nxt["cons"] = "j" if lang == "hi" else "ɲ"

    # 2. Resolve Anusvara ('M') homorganic assimilation across all 6 languages
    for idx, u in enumerate(units):
        if u["nasal"] == "M":
            nxt_raw = str(units[idx + 1]["raw_c"]) if idx + 1 < len(units) else ""
            if nxt_raw in _STOP_HOMORGANIC_NASAL:
                u["nasal"] = _STOP_HOMORGANIC_NASAL[nxt_raw]
            elif lang == "hi" and nxt_raw in ("श", "ष", "स", "ह", "य", "र", "ल", "व"):
                u["nasal"] = "̃"
            elif lang == "hi" and idx == len(units) - 1:
                # Word-final Hindi में / हैं / नहीं -> vowel nasalization
                u["nasal"] = "̃"
            else:
                # Sanskrit, Kannada, Tamil, Telugu, Malayalam word-final anusvara/sunna -> bilabial [m]
                u["nasal"] = "m"

    # 2. Hindi Ohala (1983) Schwa Deletion + 'ह' harmony
    if lang == "hi":
        # Count vowel-bearing nuclei
        nuclei = [idx for idx, u in enumerate(units) if u["vowel"]]
        if len(nuclei) >= 2:
            last_idx = nuclei[-1]
            u_last = units[last_idx]
            # Delete word-final inherent schwa if word ends on this unit (no nasal/visarga)
            # Guard: keep cluster-final schwa if preceded by bare consonant cluster like पत्र / धर्म?
            # In modern Hindi: धर्म = dʰəɾm, कर्म = kəɾm, नमस्ते = nəməsteː; keep only if C + r/j cluster like प्रिय -> pɾɪjə
            if (
                last_idx == len(units) - 1
                and u_last["inherent"]
                and not u_last["nasal"]
                and not u_last["visarga"]
            ):
                prev_is_stop_plus_tr = (
                    last_idx >= 1
                    and units[last_idx - 1]["vowel"] == ""
                    and u_last["raw_c"] in ("य",)
                )
                if not prev_is_stop_plus_tr:
                    u_last["vowel"] = ""
                    u_last["inherent"] = False

        # Medial schwa deletion: V C1(ə) C2 V_explicit -> delete ə on C1
        for idx in range(1, len(units) - 1):
            u = units[idx]
            if not u["inherent"] or u["nasal"] or u["visarga"]:
                continue
            prev_u = units[idx - 1]
            next_u = units[idx + 1]
            if prev_u["vowel"] and next_u["vowel"] and not next_u["inherent"]:
                u["vowel"] = ""
                u["inherent"] = False

    # 3. Tamil Caldwell's Law Positional Allophony
    elif lang == "ta":
        # Tamil ற்ற -> ttɾ, ன்ற -> ndɾ
        for idx in range(len(units)):
            u = units[idx]
            raw = str(u["raw_c"])
            prev_u = units[idx - 1] if idx > 0 else None
            prev_raw = str(prev_u["raw_c"]) if prev_u else ""
            prev_bare = bool(prev_u and prev_u["vowel"] == "")

            if raw == "\u0931" and prev_bare:
                if prev_raw == "\u0931":      # ற்ற -> ttɾ
                    prev_u["cons"] = "t"
                    u["cons"] = "tɾ"
                    continue
                if prev_raw in ("\u0929", "न"):  # ன்ற -> ndɾ
                    prev_u["cons"] = "n"
                    u["cons"] = "dɾ"
                    continue

            # Geminate stops stay voiceless: க்க, ச்ச, ட்ட, த்த, ப்ப
            if prev_bare and prev_raw == raw and raw in ("क", "च", "ट", "त", "प"):
                continue
            if idx + 1 < len(units) and u["vowel"] == "" and units[idx + 1]["raw_c"] == raw:
                continue

            # Post-nasal voicing: ங்க -> ŋɡ, ஞ்ச -> ɲʤ, ண்ட -> ɳɖ, ந்த -> nd, ம்ப -> mb
            prev_has_homorganic_nasal = bool(
                (prev_bare and prev_raw in ("ङ", "ञ", "ण", "न", "\u0929", "म"))
                or (prev_u and prev_u["nasal"] in ("ŋ", "ɲ", "ɳ", "n", "m"))
            )
            if prev_has_homorganic_nasal:
                voicing = {"क": "ɡ", "च": "ʤ", "ट": "ɖ", "त": "d", "प": "b"}
                if raw in voicing:
                    u["cons"] = voicing[raw]
                continue

            # Word-initial: ச -> s (standard spoken/formal Tamil சென்னை, சங்கம்)
            if idx == 0:
                if raw == "च":
                    u["cons"] = "s"
                continue

            # Intervocalic or post-liquid/approximant (V _ V or [ɾlɭɻj] _ V) lenition
            prev_voiced_sonorant = bool(
                (prev_u and prev_u["vowel"] != "")
                or (prev_bare and prev_raw in ("र", "ल", "ळ", "\u0934", "य"))
            )
            if prev_voiced_sonorant and u["vowel"] != "":
                lenition = {"क": "ɡ", "च": "s", "ट": "ɖ", "त": "ð", "प": "b"}
                if raw in lenition:
                    u["cons"] = lenition[raw]

        # Tamil non-initial unstressed 'aɪ' -> lighter [əɪ] (e.g. வானிலை -> vāniləɪ, இனிமையாக -> iniməɪyāga)
        for idx, u in enumerate(units):
            if idx > 0 and u["vowel"] == "aɪ":
                u["vowel"] = "əɪ"

    # 4. Malayalam post-nasal voicing, ന്റ -> nd, റ്റ -> tt, and word-final samvruthokaram ് -> ə
    elif lang == "ml":
        for idx, u in enumerate(units):
            raw = str(u["raw_c"])
            prev_u = units[idx - 1] if idx > 0 else None
            prev_raw = str(prev_u["raw_c"]) if prev_u else ""
            prev_bare = bool(prev_u and prev_u["vowel"] == "")

            if raw == "\u0931" and prev_bare:
                if prev_raw == "\u0931":        # റ്റ -> tt
                    prev_u["cons"] = "t"
                    u["cons"] = "t"
                elif prev_raw in ("न", "\u0d7b"):  # ന്റ -> nd
                    prev_u["cons"] = "n"
                    u["cons"] = "d"
            elif raw in ("क", "च", "ट", "त", "प"):
                is_geminate = bool(
                    (prev_bare and prev_raw == raw)
                    or (idx + 1 < len(units) and u["vowel"] == "" and units[idx + 1]["raw_c"] == raw)
                )
                prev_nasal = bool(
                    (prev_bare and prev_raw in ("ङ", "ञ", "ण", "न", "म", "\u0d7a", "\u0d7b"))
                    or (prev_u and prev_u["nasal"] in ("ŋ", "ɲ", "ɳ", "n", "m"))
                )
                if prev_nasal:
                    u["cons"] = {"क": "ɡ", "च": "ʤ", "ट": "ɖ", "त": "d", "प": "b"}[raw]
                elif not is_geminate and idx > 0 and u["vowel"] != "" and prev_u and prev_u["vowel"] != "":
                    # Malayalam intervocalic dental/retroflex stop voicing (e.g. ഭാഷയുടെ -> bʰɑːʂɐjʊɖe, സ്വാഗതം -> sʋɑːɡɐdəm)
                    if raw in ("ट", "त"):
                        u["cons"] = {"ट": "ɖ", "त": "d"}[raw]

        # Word-final chandrakkala (samvruthokaram, e.g. ഇന്ന്, ആണ്) -> clean central schwa release ə
        if units[-1]["final_virama"] and units[-1]["raw_c"] not in _ML_CHILLU_IPA:
            units[-1]["vowel"] = "ə"

    # 5. Assign natural primary word stress ('ˈ') if no explicit Vedic svara stress was marked
    if not any(bool(u["stress"]) for u in units):
        vowel_indices = [idx for idx, u in enumerate(units) if u["vowel"]]
        if vowel_indices:
            chosen = vowel_indices[0]
            # Prefer long-vowel heavy syllable first (e.g. नमस्कारम् -> kˈɑː, संश्लेषण -> lˈeː)
            found_long = False
            for v_idx in vowel_indices:
                if str(units[v_idx]["vowel"]) in _LONG_VOWEL_IPA:
                    chosen = v_idx
                    found_long = True
                    break
            if not found_long:
                for v_idx in vowel_indices:
                    u = units[v_idx]
                    is_closed = (
                        bool(u["nasal"])
                        or bool(u["visarga"])
                        or (v_idx + 1 < len(units) and units[v_idx + 1]["vowel"] == "")
                    )
                    if is_closed:
                        chosen = v_idx
                        break
            units[chosen]["stress"] = True

    # 6. Stressed central vowel ('ˈʌ' instead of out-of-distribution 'ˈə') + Medial anti-elision 'ɐ' for non-Hindi
    for idx, u in enumerate(units):
        if u["stress"] and u["vowel"] == "ə":
            u["vowel"] = "ʌ"
        elif u["stress"] and u["vowel"] == "əɪ":
            u["vowel"] = "ʌɪ"
        elif u["stress"] and u["vowel"] == "əʊ":
            u["vowel"] = "ʌʊ"
        elif (
            lang != "hi"
            and 0 < idx < len(units) - 1
            and u["inherent"]
            and u["vowel"] == "ə"
            and not u["stress"]
            and not u["nasal"]
            and not u["visarga"]
        ):
            u["vowel"] = "ɐ"


def _render_units_to_ipa(units: list[dict[str, object]], lang: str) -> str:
    parts: list[str] = []
    for idx, u in enumerate(units):
        cons = str(u["cons"])
        vowel = str(u["vowel"])
        nasal = str(u["nasal"])
        visarga = bool(u["visarga"])
        stress = bool(u["stress"])

        if stress and vowel:
            parts.append(cons + "ˈ" + vowel)
        else:
            parts.append(cons + vowel)

        if nasal:
            parts.append(nasal)

        if visarga:
            nxt_raw = str(units[idx + 1]["raw_c"]) if idx + 1 < len(units) else ""
            if nxt_raw in ("च", "छ"):
                parts.append("ɕ")
            elif nxt_raw in ("ट", "ठ"):
                parts.append("ʂ")
            elif nxt_raw in ("त", "थ"):
                parts.append("s")
            elif idx == len(units) - 1 and lang == "sa" and vowel:
                # Classical Sanskrit clause/word-final visarga breath echo (e.g. युयुत्सवः -> əhᵊ, मामकाः -> ɑːh)
                if vowel == "ə":
                    parts.append("hᵊ")
                elif vowel == "ɪ":
                    parts.append("hɪ")
                elif vowel == "ʊ":
                    parts.append("hʊ")
                else:
                    parts.append("h")
            else:
                parts.append("h")

    return "".join(parts)


_DIGIT_WORDS_HI = {
    "0": "ʃˈuːnjə", "1": "ˈeːk", "2": "dˈoː", "3": "tˈiːn", "4": "ʧˈɑːɾ",
    "5": "pˈɑ̃ːʧ", "6": "ʧʰˈɛː", "7": "sˈɑːt", "8": "ˈɑːʈʰ", "9": "nˈɔː",
}
_INDIC_ZERO_BASES = (0x0966, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66)
_LATIN_CHAR_IPA = {
    "a": "ə", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ",
    "h": "h", "i": "ɪ", "j": "ʤ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "k", "r": "ɾ", "s": "s", "t": "t", "u": "ʊ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z",
}


def _latin_word_to_ipa(word: str) -> str:
    w = word.lower()
    for dg, rep in (("tion", "ʃən"), ("sh", "ʃ"), ("ch", "ʧ"), ("th", "tʰ"), ("ph", "f"), ("ng", "ŋ"), ("ee", "iː"), ("oo", "uː")):
        w = w.replace(dg, rep)
    out: list[str] = []
    for ch in w:
        if ch in _LATIN_CHAR_IPA:
            out.append(_LATIN_CHAR_IPA[ch])
        elif ch in IPA_VOCAB:
            out.append(ch)
    return "".join(out)


def _normalize_digits(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        replaced = False
        for base in _INDIC_ZERO_BASES:
            if base <= cp <= base + 9:
                out.append(str(cp - base))
                replaced = True
                break
        if not replaced:
            out.append(ch)
    return "".join(out)


def text_to_ipa(text: str, lang: str = "sa", normalized_dev_fallback: str | None = None) -> str:
    """Convert any Sanskrit / Hindi / Kannada / Tamil / Telugu / Malayalam / IAST text into exact 24 kHz IPA."""
    raw = _normalize_digits(str(text or "")).strip()
    if not raw:
        return "ə"

    # Normalize dandas and ellipses into clean prosodic punctuation:
    # Double danda -> '.', Hindi/clause-final single danda -> '.', mid-sloka single danda -> ';'
    raw = raw.replace("\u0965", ".").replace("||", ".")
    if lang == "sa" and ("\u0964" in raw or "|" in raw):
        raw = re.sub(r"[\u0964|]\s*$", ".", raw)
        raw = raw.replace("\u0964", ";").replace("|", ";")
    else:
        raw = raw.replace("\u0964", ".").replace("|", ".")
    raw = raw.replace("...", "…").replace("–", "—")

    # If input was Roman/IAST and caller provided normalized Devanagari, use that
    if normalized_dev_fallback and re.search(r"[A-Za-z\u0100-\u024F]", raw):
        work = normalized_dev_fallback.replace("\u0965", ".").replace("\u0964", ",")
    else:
        work = raw

    tokens = re.findall(r"[^\s,.;:!?—…\"()“”]+|[,.;:!?—…\"()“”]+|\s+", work)
    out_chunks: list[str] = []

    for tok in tokens:
        if tok.isspace():
            out_chunks.append(" ")
            continue
        if re.fullmatch(r"[,.;:!?—…\"()“”]+", tok):
            p = tok[0]
            # Sentence-final vowel release guard: prevent punctuation token from clipping preceding vowel/nasal tail
            if out_chunks and out_chunks[-1] and out_chunks[-1][-1] in _RELEASE_TAIL_CHARS and p in (".", ",", "!", "?", ";", ":"):
                out_chunks.append(" " + p)
            else:
                out_chunks.append(p)
            continue
        if tok.isdigit():
            out_chunks.append(" ".join(_DIGIT_WORDS_HI.get(d, "ʃˈuːnjə") for d in tok))
            continue
        if re.fullmatch(r"[A-Za-z']+", tok):
            lat_ipa = _latin_word_to_ipa(tok)
            if lat_ipa:
                out_chunks.append(lat_ipa)
            continue

        dev_word = _normalize_script_word_to_dev(tok, lang=lang)
        units = _parse_word_units(dev_word, lang=lang)
        if not units:
            continue
        _apply_language_allophony(units, lang=lang)
        ipa_word = _render_units_to_ipa(units, lang=lang)
        if ipa_word:
            out_chunks.append(ipa_word)

    ipa = re.sub(r"\s+", " ", "".join(out_chunks)).strip()
    # Filter strictly to valid vocabulary symbols
    filtered = "".join(ch for ch in ipa if ch in IPA_VOCAB)
    if not filtered:
        return "ə"
    # If clause ends on a bare vowel/length/nasal without punctuation, append soft period with vowel guard space
    if filtered[-1] in _RELEASE_TAIL_CHARS:
        filtered = filtered + " ."
    # Pause-final bare 'e' (without 'ː' or 'ɪ') is unvoiced in StyleTTS2/espeak; lower to open-mid 'ɛ' before pause
    filtered = re.sub(r"e (?=[.,!?;:])", "ɛ ", filtered)
    return filtered


def ipa_to_ids(ipa: str, max_tokens: int = 510) -> list[int]:
    """Map an IPA string to `[0, ...token_ids..., 0]` bounded safely to PL-BERT's 512 context window."""
    core = [IPA_VOCAB[ch] for ch in ipa if ch in IPA_VOCAB]
    if not core:
        core = [IPA_VOCAB["ə"]]
    if len(core) > max_tokens:
        core = core[:max_tokens]
    return [0, *core, 0]
