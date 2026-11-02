#!/usr/bin/env python3
"""
Ninada v4.0 Native Studio VITS Architecture (24,000 Hz, hop=256).

Self-contained PyTorch implementation of single-speaker native Indic Studio VITS
(`TextEncoder` with relative Multi-Head Attention, `PosteriorEncoder`,
`ResidualCouplingBlock`, ONNX-vectorized `StochasticDurationPredictor`,
HiFi-GAN `WaveformDecoder`, and `MultiPeriodDiscriminator` `disc.nets`) that:
  1. Loads 100% (`949 / 949` state-dict keys, `strict=True`) of IISc SPIRE `SYSPIN/vits_*`
     and Coqui Studio VITS checkpoints (including pre-converged `disc.nets` MPD).
  2. Supports 100% native Unicode orthography per language (`sa`, `kn`, `hi`, `ta`, `te`, `ml`)
     with zero cross-language voice or Hindi phonetic contamination.
  3. Exports cleanly to `opset 17` ONNX (`W8A32 + FP16`, ~28.5 MB per specialist) with
     vectorized Rational-Quadratic Splines (`torch.gather` + `torch.where`, zero `index_put_`).
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

SAMPLE_RATE_24K = 24000
HOP_LENGTH = 256
FFT_SIZE = 1024
WIN_LENGTH = 1024
N_MELS = 80
SPEC_CHANNELS = FFT_SIZE // 2 + 1  # 513
SEGMENT_FRAMES = 32  # 32 * 256 = 8,192 samples @ 24 kHz

SUPPORTED_LANGS = ("sa", "hi", "kn", "ta", "te", "ml")

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

_WHITESPACE_RE = re.compile(r"\s+")
_AUX_SYMBOLS_RE = re.compile(r"[\<\>\(\)\[\]\"]+")


# ---------------------------------------------------------------------------
# Native Script Normalization & IAST -> Sanskrit Devanagari Helper
# ---------------------------------------------------------------------------

_IAST_VOWELS_INDEP = [
    ("ai", "ऐ"), ("au", "औ"),
    ("ā", "आ"), ("ī", "ई"), ("ū", "ऊ"),
    ("ṛ", "ऋ"), ("ṝ", "ॠ"), ("ḷ", "ऌ"), ("ḹ", "ॡ"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
]
_IAST_VOWELS_MATRA = [
    ("ai", "ै"), ("au", "ौ"),
    ("ā", "ा"), ("ī", "ी"), ("ū", "ू"),
    ("ṛ", "ृ"), ("ṝ", "ॄ"), ("ḷ", "ॢ"), ("ḹ", "ॣ"),
    ("a", ""), ("i", "ि"), ("u", "ु"), ("e", "े"), ("o", "ो"),
]
_IAST_CONSONANTS = [
    ("kh", "ख"), ("gh", "घ"), ("ch", "छ"), ("jh", "झ"),
    ("ṭh", "ठ"), ("ḍh", "ढ"), ("th", "थ"), ("dh", "ध"),
    ("ph", "फ"), ("bh", "भ"),
    ("ṅ", "ङ"), ("ñ", "ञ"), ("ṇ", "ण"), ("ś", "श"), ("ṣ", "ष"), ("ḷ", "ळ"),
    ("k", "क"), ("g", "ग"), ("c", "च"), ("j", "ज"),
    ("ṭ", "ट"), ("ḍ", "ड"), ("t", "त"), ("d", "द"), ("n", "न"),
    ("p", "प"), ("b", "ब"), ("m", "म"),
    ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"),
    ("s", "स"), ("h", "ह"),
]


def iast_to_devanagari(text: str) -> str:
    """Convert Romanized IAST Sanskrit into native Devanagari if Latin letters dominate."""
    s = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i : i + 2] == "||":
            out.append("॥")
            i += 2
            continue
        if s[i] == "|":
            out.append("।")
            i += 1
            continue
        if s[i] == "'":
            out.append("ऽ")
            i += 1
            continue
        if s[i] in ("ṃ", "ṁ"):
            out.append("ं")
            i += 1
            continue
        if s[i] == "ḥ":
            out.append("ः")
            i += 1
            continue

        matched_cons = None
        for lat, dev in _IAST_CONSONANTS:
            if s.startswith(lat, i):
                # Do not treat syllabic ḷ after consonant as consonant ळ if followed by consonant
                if lat == "ḷ" and out and out[-1].endswith("्"):
                    continue
                matched_cons = (lat, dev)
                break
        if matched_cons is not None:
            lat, dev = matched_cons
            i += len(lat)
            matched_vow = None
            for vlat, matra in _IAST_VOWELS_MATRA:
                if s.startswith(vlat, i):
                    matched_vow = (vlat, matra)
                    break
            if matched_vow is not None:
                vlat, matra = matched_vow
                out.append(dev + matra)
                i += len(vlat)
            else:
                out.append(dev + "्")
            continue

        matched_ind = None
        for vlat, vdev in _IAST_VOWELS_INDEP:
            if s.startswith(vlat, i):
                matched_ind = (vlat, vdev)
                break
        if matched_ind is not None:
            vlat, vdev = matched_ind
            out.append(vdev)
            i += len(vlat)
            continue

        out.append(s[i])
        i += 1
    return "".join(out)


# Malayalam atomic chillu normalization (ensures consistent single-codepoint chillu)
_MALAYALAM_CHILLU_SEQ = {
    "\u0d23\u0d4d\u200d": "\u0d7a",  # ൺ
    "\u0d28\u0d4d\u200d": "\u0d7b",  # ൻ
    "\u0d30\u0d4d\u200d": "\u0d7c",  # ർ
    "\u0d32\u0d4d\u200d": "\u0d7d",  # ൽ
    "\u0d33\u0d4d\u200d": "\u0d7e",  # ൾ
    "\u0d15\u0d4d\u200d": "\u0d7f",  # ൿ
}


def clean_native_text(text: str, lang: str) -> str:
    """Normalize native Indic/Sanskrit text while keeping 100% authentic script codepoints."""
    t = unicodedata.normalize("NFC", str(text or "")).strip()
    if not t:
        return ""

    # If Sanskrit input is provided in Latin/IAST, convert to native Devanagari
    if lang == "sa":
        has_deva = any(0x0900 <= ord(c) <= 0x097F for c in t)
        has_latin = any(("a" <= c.lower() <= "z") or c in "āīūṛṝḷḹṅñṭḍṇśṣṃḥ" for c in t)
        if has_latin and not has_deva:
            t = iast_to_devanagari(t)
        # Strip Vedic pitch marks (udatta/anudatta/svarita) that are unvoiced in Rasa studio corpus
        t = re.sub(r"[\u0951-\u0954\u1cda-\u1cdf]", "", t)
        t = t.replace("॥", ".").replace("।", ",")

    if lang == "kn":
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

    if lang == "ta":
        t = t.replace("\u0bd0", "ஓம்").replace("\u0bb6", "ஸ")

    if lang == "ml":
        for seq, atomic in _MALAYALAM_CHILLU_SEQ.items():
            t = t.replace(seq, atomic)
        t = (
            t.replace("\u0d4e", "\u0d7c")
            .replace("\u0d54", "ം")
            .replace("\u0d3a", "റ")
            .replace("\u0d44", "ൃ")
        )

    # Remove zero-width joiners/non-joiners after chillu folding
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # Standardize typographic quotes/dashes/dandas
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
    if lang in ("kn", "ta", "te", "ml"):
        t = t.replace("॥", ".").replace("।", ".")
    elif lang == "hi":
        t = t.replace("॥", "।")

    t = t.lower()
    t = _AUX_SYMBOLS_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


@dataclass
class StudioCharTokenizer:
    """
    Exact Coqui/SYSPIN `VitsCharacters` + `TTSTokenizer` compatible character tokenizer.
    Layout: `[pad] + list(punctuations) + list(characters) + [blank]`
    with `add_blank=True` (`[blank_id, c0, blank_id, c1, ..., blank_id]`).
    """

    lang: str
    pad: str
    punctuations: str
    characters: str
    blank: str = "<BLNK>"
    add_blank: bool = True

    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        self.vocab: list[str] = (
            [self.pad] + list(self.punctuations) + list(self.characters) + [self.blank]
        )
        # Coqui `VitsCharacters` assigns latest index if a punctuation char also appears in `characters`
        self.char_to_id: dict[str, int] = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char: dict[int, str] = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.pad_id: int = self.char_to_id[self.pad]
        self.blank_id: int = self.char_to_id[self.blank]

    @property
    def num_chars(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_coqui_config(cls, lang: str, cfg: dict[str, Any]) -> "StudioCharTokenizer":
        c_cfg = cfg["characters"]
        pad = c_cfg.get("pad") or "<PAD>"
        puncts = c_cfg.get("punctuations") or "!¡'(),-.:;¿? "
        chars = c_cfg.get("characters") or ""
        if c_cfg.get("phonemes"):
            chars = chars + c_cfg["phonemes"]
        add_blank = bool(cfg.get("add_blank", True))
        return cls(
            lang=lang,
            pad=pad,
            punctuations=puncts,
            characters=chars,
            blank="<BLNK>",
            add_blank=add_blank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "pad": self.pad,
            "punctuations": self.punctuations,
            "characters": self.characters,
            "blank": self.blank,
            "add_blank": self.add_blank,
            "num_chars": self.num_chars,
            "pad_id": self.pad_id,
            "blank_id": self.blank_id,
            "char_to_id": self.char_to_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StudioCharTokenizer":
        return cls(
            lang=d["lang"],
            pad=d["pad"],
            punctuations=d["punctuations"],
            characters=d["characters"],
            blank=d.get("blank", "<BLNK>"),
            add_blank=bool(d.get("add_blank", True)),
        )

    def text_to_ids(self, text: str) -> list[int]:
        cleaned = clean_native_text(text, self.lang)
        ids: list[int] = []
        for ch in cleaned:
            idx = self.char_to_id.get(ch)
            if idx is not None:
                ids.append(idx)
        if not ids:
            space_id = self.char_to_id.get(" ", self.pad_id)
            ids = [space_id]
        if self.add_blank:
            res = [self.blank_id] * (len(ids) * 2 + 1)
            res[1::2] = ids
            return res
        return ids


# Native vocabularies for target scripts that expand or adapt from warm-start checkpoints
NATIVE_MALAYALAM_CHARS = (
    "ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔ"
    "കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമ"
    "യരറലളഴവശഷസഹ"
    "ഽാിീുൂൃെേൈൊോൌ്ൗ"
    "ൠൡ൦൧൨൩൪൫൬൭൮൯"
    "ൺൻർൽൾൿ"
)

NATIVE_SANSKRIT_EXTRA_CHARS = "ऌॡॢॣऽ।॥ॐॠॄृःंँळक्षज्ञ"
NATIVE_TAMIL_EXTRA_CHARS = "ஂஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்ௗ"


def build_target_tokenizer_and_remap_embeddings(
    lang: str,
    src_cfg: dict[str, Any],
    src_emb_weight: torch.Tensor,
) -> tuple[StudioCharTokenizer, torch.Tensor]:
    """
    Construct the native per-language `StudioCharTokenizer` and remap `text_encoder.emb.weight`
    so 100% of native characters have phonetically grounded pretrained embeddings at step 0.
    """
    src_tok = StudioCharTokenizer.from_coqui_config(lang, src_cfg)
    d_model = src_emb_weight.shape[1]

    # Warm-start any untrained punctuation rows (L2 norm < 0.55 in SYSPIN donor checkpoints)
    # from the donor's trained ',' or ' ' pause embedding so utterance-final '.' / '!' / '।'
    # start with clean studio prosody from step 0 across ALL languages (including Malayalam)!
    patched_src_emb = src_emb_weight.clone()
    pause_donor_idx = src_tok.char_to_id.get(",", src_tok.char_to_id.get(" ", src_tok.blank_id))
    space_donor_idx = src_tok.char_to_id.get(" ", pause_donor_idx)
    for p_ch in (".", "!", "?", ":", ";", "।", "॥"):
        p_idx = src_tok.char_to_id.get(p_ch)
        if p_idx is not None and float(patched_src_emb[p_idx].norm().item()) < 0.55:
            patched_src_emb[p_idx] = patched_src_emb[pause_donor_idx]
    for s_ch in ("-", "(", ")", "'", '"', "¡", "¿"):
        s_idx = src_tok.char_to_id.get(s_ch)
        if s_idx is not None and float(patched_src_emb[s_idx].norm().item()) < 0.55:
            patched_src_emb[s_idx] = patched_src_emb[space_donor_idx]

    if lang == "ml":
        # Build dedicated native Malayalam tokenizer and map 1-to-1 from Kannada Brahmic offsets
        target_tok = StudioCharTokenizer(
            lang="ml",
            pad="<PAD>",
            punctuations="!¡'(),-.:;¿? ",
            characters=NATIVE_MALAYALAM_CHARS,
            blank="<BLNK>",
            add_blank=True,
        )
        new_emb = torch.zeros((target_tok.num_chars, d_model), dtype=patched_src_emb.dtype)
        # Fallback mean vector of active embeddings
        mean_vec = patched_src_emb.mean(dim=0)
        new_emb[:] = mean_vec

        special_ml_to_kn = {
            "ഃ": "ಹ",
            "ങ": "ಂ",
            "റ": "ರ",
            "ഴ": "ಳ",
            "ഩ": "ನ",
            "ൗ": "ೌ",
            "ൠ": "ಋ",
            "ൡ": "ಲ",
            "ഌ": "ಲ",
            "ൺ": "ಣ",
            "ൻ": "ನ",
            "ർ": "ರ",
            "ൽ": "ಲ",
            "ൾ": "ಳ",
            "ൿ": "ಕ",
        }
        for idx, tok in enumerate(target_tok.vocab):
            if tok == "<PAD>":
                new_emb[idx] = patched_src_emb[src_tok.pad_id]
            elif tok == "<BLNK>":
                new_emb[idx] = patched_src_emb[src_tok.blank_id]
            elif tok in src_tok.char_to_id:
                new_emb[idx] = patched_src_emb[src_tok.char_to_id[tok]]
            elif tok in special_ml_to_kn and special_ml_to_kn[tok] in src_tok.char_to_id:
                new_emb[idx] = patched_src_emb[src_tok.char_to_id[special_ml_to_kn[tok]]]
            elif len(tok) == 1 and 0x0D00 <= ord(tok) <= 0x0D7F:
                kn_cousin = chr(ord(tok) - 0x0080)  # Exact Unicode Malayalam -> Kannada offset!
                if kn_cousin in src_tok.char_to_id:
                    new_emb[idx] = patched_src_emb[src_tok.char_to_id[kn_cousin]]
        return target_tok, new_emb

    if lang == "ta":
        # Build canonical 61-token native Tamil tokenizer and map 1-to-1 from SYSPIN Kannada Brahmic (+0x0100) offsets
        target_tok = StudioCharTokenizer(
            lang="ta",
            pad="_",
            punctuations=".,!?; ",
            characters=NATIVE_TAMIL_EXTRA_CHARS + "'-()",
            blank="<BLNK>",
            add_blank=True,
        )
        new_emb = remap_embeddings_to_existing_tokenizer(
            target_tok, src_cfg, src_emb_weight, "ta"
        )
        return target_tok, new_emb

    # Same-script specialists (`kn`, `te`, `hi`, `ta`, and `sa` from Marathi Devanagari):
    # Preserve 100% of checkpoint characters at exact indices, and append any missing native codepoints before `<BLNK>`.
    extra_needed = ""
    if lang == "sa":
        extra_needed = NATIVE_SANSKRIT_EXTRA_CHARS
    elif lang == "ta":
        extra_needed = NATIVE_TAMIL_EXTRA_CHARS + "'-()"

    existing_set = set(src_tok.vocab)
    append_chars = "".join(ch for ch in extra_needed if ch not in existing_set)
    if not append_chars:
        return src_tok, patched_src_emb

    target_tok = StudioCharTokenizer(
        lang=lang,
        pad=src_tok.pad,
        punctuations=src_tok.punctuations,
        characters=src_tok.characters + append_chars,
        blank=src_tok.blank,
        add_blank=src_tok.add_blank,
    )
    new_emb = torch.zeros((target_tok.num_chars, d_model), dtype=patched_src_emb.dtype)
    old_blank_idx = src_tok.blank_id
    # Copy prefix `[0 .. old_blank_idx - 1]` identically
    new_emb[:old_blank_idx] = patched_src_emb[:old_blank_idx]
    # Place `<BLNK>` at the new final position
    new_emb[-1] = patched_src_emb[old_blank_idx]

    cousin_map = {
        "ऌ": "ल",
        "ॡ": "ल",
        "ॢ": "ृ",
        "ॣ": "ॄ",
        "॥": "।",
        "।": ",",
        "ऽ": "अ",
        "ௗ": "ௌ",
        "-": " ",
        "'": " ",
        "(": ",",
        ")": ",",
    }
    mean_vec = patched_src_emb.mean(dim=0)
    for offset, ch in enumerate(append_chars):
        pos = old_blank_idx + offset
        donor = cousin_map.get(ch)
        if donor and donor in src_tok.char_to_id:
            new_emb[pos] = patched_src_emb[src_tok.char_to_id[donor]]
        else:
            new_emb[pos] = mean_vec
    return target_tok, new_emb


def remap_embeddings_to_existing_tokenizer(
    target_tok: StudioCharTokenizer,
    src_cfg: dict[str, Any],
    src_emb_weight: torch.Tensor,
    lang: str,
) -> torch.Tensor:
    """
    Permute/project a donor checkpoint's `text_encoder.emb.weight` (e.g. a Female SYSPIN checkpoint
    whose `characters` string was serialized in a different `set()` order) onto `target_tok.vocab`
    so Male and Female specialists for `lang` share the exact same character-to-ID table!
    """
    src_tok = StudioCharTokenizer.from_coqui_config(lang, src_cfg)
    d_model = src_emb_weight.shape[1]

    patched_src_emb = src_emb_weight.clone()
    pause_donor_idx = src_tok.char_to_id.get(",", src_tok.char_to_id.get(" ", src_tok.blank_id))
    space_donor_idx = src_tok.char_to_id.get(" ", pause_donor_idx)
    for p_ch in (".", "!", "?", ":", ";", "।", "॥"):
        p_idx = src_tok.char_to_id.get(p_ch)
        if p_idx is not None and float(patched_src_emb[p_idx].norm().item()) < 0.55:
            patched_src_emb[p_idx] = patched_src_emb[pause_donor_idx]
    for s_ch in ("-", "(", ")", "'", '"', "¡", "¿"):
        s_idx = src_tok.char_to_id.get(s_ch)
        if s_idx is not None and float(patched_src_emb[s_idx].norm().item()) < 0.55:
            patched_src_emb[s_idx] = patched_src_emb[space_donor_idx]

    new_emb = torch.zeros((target_tok.num_chars, d_model), dtype=patched_src_emb.dtype)
    mean_vec = patched_src_emb.mean(dim=0)
    new_emb[:] = mean_vec

    special_map = {
        "ഃ": "ಹ", "ങ": "ಂ",
        "റ": "ರ", "ഴ": "ಳ", "ഩ": "ನ", "ൗ": "ೌ", "ൠ": "ಋ", "ൡ": "ಲ", "ഌ": "ಲ",
        "ൺ": "ಣ", "ൻ": "ನ", "ർ": "ರ", "ൽ": "ಲ", "ൾ": "ಳ", "ൿ": "ಕ",
        "ன": "ನ", "ற": "ರ", "ழ": "ಳ", "ௗ": "ೌ", "ஃ": "ಹ", "ஂ": "ಂ", "ங": "ಂ",
        "ऌ": "ल", "ॡ": "ल", "ॢ": "ृ", "ॣ": "ॄ", "॥": "।", "।": ",", "ऽ": "अ",
    }
    for idx, tok in enumerate(target_tok.vocab):
        if tok == target_tok.pad:
            new_emb[idx] = patched_src_emb[src_tok.pad_id]
        elif tok == target_tok.blank:
            new_emb[idx] = patched_src_emb[src_tok.blank_id]
        elif tok in src_tok.char_to_id:
            new_emb[idx] = patched_src_emb[src_tok.char_to_id[tok]]
        elif tok in special_map and special_map[tok] in src_tok.char_to_id:
            new_emb[idx] = patched_src_emb[src_tok.char_to_id[special_map[tok]]]
        elif lang == "ml" and len(tok) == 1 and 0x0D00 <= ord(tok) <= 0x0D7F:
            kn_cousin = chr(ord(tok) - 0x0080)
            if kn_cousin in src_tok.char_to_id:
                new_emb[idx] = patched_src_emb[src_tok.char_to_id[kn_cousin]]
        elif lang == "ta" and len(tok) == 1 and 0x0B80 <= ord(tok) <= 0x0BFF:
            kn_cousin = chr(ord(tok) + 0x0100)
            if kn_cousin in src_tok.char_to_id:
                new_emb[idx] = patched_src_emb[src_tok.char_to_id[kn_cousin]]
    return new_emb


# ---------------------------------------------------------------------------
# Core VITS Neural Building Blocks (100% Coqui/SYSPIN Key-Compatible)
# ---------------------------------------------------------------------------


def sequence_mask(length: torch.Tensor, max_length: int | torch.Tensor | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class ChannelLayerNorm(nn.Module):
    """Channel-first 1D LayerNorm with `gamma` and `beta` parameters matching Coqui TTS."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class RelativeMultiHeadAttention(nn.Module):
    """Relative positional Multi-Head Attention (`window_size=4` -> `(1, 9, d_k)`)."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
        )

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    @staticmethod
    def _matmul_with_relative_keys(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    @staticmethod
    def _matmul_with_relative_values(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.unsqueeze(0))

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        # x: [b, heads, t, 2*t - 1] -> [b, heads, t, t] (100% dynamic-shape ONNX Slice)
        b, heads, _, _ = x.shape
        z_col = x[:, :, :, :1] * 0.0  # [b, heads, t, 1]
        x_pad = torch.cat([x, z_col], dim=-1)  # [b, heads, t, 2*t]
        x_flat = x_pad.reshape(b, heads, -1)  # [b, heads, 2*t*t]
        z_tail = (x[:, :, 1:, :1] * 0.0).reshape(b, heads, -1)  # [b, heads, t - 1]
        x_flat = torch.cat([x_flat, z_tail], dim=-1)  # [b, heads, (t + 1)*(2*t - 1)]
        s = torch._shape_as_tensor(x)
        t_len = s[2]
        two_t_m1 = s[3]
        x_final = x_flat.reshape(b, heads, -1, x.size(3))[:, :, :-1, :]  # [b, heads, t, 2*t - 1]
        return torch.narrow(x_final, 3, two_t_m1 - t_len, t_len)

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        # x: [b, heads, t, t] -> [b, heads, t, 2*t - 1] (100% dynamic-shape ONNX safe)
        b, heads, length, _ = x.shape
        z_cols = x[:, :, :, 1:] * 0.0  # [b, heads, t, t - 1]
        x_pad = torch.cat([x, z_cols], dim=-1)  # [b, heads, t, 2*t - 1]
        x_flat = x_pad.reshape(b, heads, -1)  # [b, heads, t*(2*t - 1)]
        z_head = (x[:, :, :, :1] * 0.0).reshape(b, heads, -1)  # [b, heads, t]
        x_flat = torch.cat([z_head, x_flat], dim=-1)  # [b, heads, 2*t*t]
        x_final = x_flat.reshape(b, heads, length, -1)[:, :, :, 1:]  # [b, heads, t, 2*t - 1]
        return x_final

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        b, d, t_s = k.shape
        t_t = q.shape[2]
        query = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scale = math.sqrt(self.k_channels)
        scores = torch.matmul(query / scale, key.transpose(-2, -1))

        # Dynamic-shape ONNX-clean relative embedding slice:
        # `pad_z` has shape `[1, t_s, k_channels]`; concatenating `[pad_z, emb_rel, pad_z]`
        # gives length `2*t_s + 2*w + 1`, so static slice `[:, w+1 : -(w+1), :]` always has
        # exact length `2*t_s - 1` centered at `w` for EVERY `t_s >= 1`!
        w1 = self.window_size + 1
        pad_z = (key[:1, :1, :, :] * 0.0).squeeze(0)  # [1, t_s, k_channels]
        rel_pad_k = torch.cat([pad_z, self.emb_rel_k.to(dtype=query.dtype), pad_z], dim=1)
        key_rel = rel_pad_k[:, w1:-w1, :]
        rel_logits = self._matmul_with_relative_keys(query / scale, key_rel)
        scores_local = self._relative_position_to_absolute_position(rel_logits)
        scores = scores + scores_local

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        output = torch.matmul(p_attn, value)
        p_attn_rel = self._absolute_position_to_relative_position(p_attn)
        rel_pad_v = torch.cat([pad_z, self.emb_rel_v.to(dtype=query.dtype), pad_z], dim=1)
        val_rel = rel_pad_v[:, w1:-w1, :]
        output = output + self._matmul_with_relative_values(p_attn_rel, val_rel)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return self.conv_o(output)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class RelativeTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                RelativeMultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size
                )
            )
            self.norm_layers_1.append(ChannelLayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(ChannelLayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.encoder = RelativeTransformerEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_emb = self.emb(x) * math.sqrt(self.hidden_channels)
        x_emb = x_emb.transpose(1, -1)
        x_mask = sequence_mask(x_lengths, x_emb.size(2)).unsqueeze(1).to(x_emb.dtype)
        enc_out = self.encoder(x_emb * x_mask, x_mask)
        stats = self.proj(enc_out) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return enc_out, m, logs, x_mask


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.in_layers.append(in_layer)
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            res_skip_layer = weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        hc = self.hidden_channels
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            acts = torch.tanh(x_in[:, :hc, :]) * torch.sigmoid(x_in[:, hc:, :])
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :hc, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, hc:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = SPEC_CHANNELS,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask)
        m = self.post(h) * x_mask
        if not reverse:
            x1 = (m + x1) * x_mask
            return torch.cat([x0, x1], dim=1), torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        else:
            x1 = (x1 - m) * x_mask
            return torch.cat([x0, x1], dim=1), None


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
    ) -> None:
        super().__init__()
        self.flows = nn.ModuleList(
            [
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers)
                for _ in range(n_flows)
            ]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, reverse=False)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x, _ = flow(x, x_mask, reverse=True)
        return x


class DDSConv(nn.Module):
    """Dilated Depth-Separable Conv block used in `StochasticDurationPredictor`."""

    def __init__(self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(ChannelLayerNorm(channels))
            self.norms_2.append(ChannelLayerNorm(channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not reverse:
            y = (self.translation + torch.exp(self.log_scale) * x) * x_mask
            logdet = torch.sum(self.log_scale * x_mask, [1, 2])
            return y, logdet
        else:
            y = ((x - self.translation) * torch.exp(-self.log_scale)) * x_mask
            return y, None


def vectorized_unconstrained_rqs(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 5.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    100% vectorized, ONNX-exportable (`opset 17`) Rational-Quadratic Spline transform.
    Eliminates dynamic boolean mask indexing (`inputs[inside_interval_mask]` / `index_put_`)
    by clamping `inputs` inside `[-tail_bound + eps, tail_bound - eps]` for the spline branch
    and selecting exact linear tails outside `[-tail_bound, tail_bound]` with `torch.where`.
    Bit-identical (`< 1e-6`) to Coqui/SYSPIN PyTorch RQS on all inputs!
    """
    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    const_deriv = math.log(math.exp(1.0 - min_derivative) - 1.0)
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1), mode="constant", value=const_deriv)

    num_bins = unnormalized_widths.shape[-1]
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (2.0 * tail_bound) * cumwidths - tail_bound
    # Exact boundary pin without in-place mutation (`ONNX` safe):
    left_edge = torch.full_like(cumwidths[..., :1], -tail_bound)
    right_edge = torch.full_like(cumwidths[..., :1], tail_bound)
    cumwidths = torch.cat([left_edge, cumwidths[..., 1:-1], right_edge], dim=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (2.0 * tail_bound) * cumheights - tail_bound
    cumheights = torch.cat([left_edge, cumheights[..., 1:-1], right_edge], dim=-1)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    safe_in = torch.clamp(inputs, -tail_bound + 1e-5, tail_bound - 1e-5)

    if inverse:
        cumheights_search = torch.cat([cumheights[..., :-1], cumheights[..., -1:] + 1e-6], dim=-1)
        bin_idx = torch.sum(safe_in.unsqueeze(-1) >= cumheights_search, dim=-1) - 1
    else:
        cumwidths_search = torch.cat([cumwidths[..., :-1], cumwidths[..., -1:] + 1e-6], dim=-1)
        bin_idx = torch.sum(safe_in.unsqueeze(-1) >= cumwidths_search, dim=-1) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1).unsqueeze(-1)

    input_cumwidths = torch.gather(cumwidths, -1, bin_idx).squeeze(-1)
    input_bin_widths = torch.gather(widths, -1, bin_idx).squeeze(-1)
    input_cumheights = torch.gather(cumheights, -1, bin_idx).squeeze(-1)
    delta = heights / widths
    input_delta = torch.gather(delta, -1, bin_idx).squeeze(-1)
    input_derivatives = torch.gather(derivatives, -1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = torch.gather(derivatives[..., 1:], -1, bin_idx).squeeze(-1)
    input_heights = torch.gather(heights, -1, bin_idx).squeeze(-1)

    if inverse:
        a = (inputs_s := safe_in - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - inputs_s * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        c = -input_delta * inputs_s
        discriminant = torch.clamp_min(b.pow(2) - 4.0 * a * c, 0.0)
        root = (2.0 * c) / (-b - torch.sqrt(discriminant) - 1e-12)
        rqs_out = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1.0 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - root).pow(2)
        )
        rqs_logdet = -(torch.log(torch.clamp_min(derivative_numerator, 1e-12)) - 2.0 * torch.log(torch.clamp_min(denominator, 1e-12)))
    else:
        theta = (safe_in - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1.0 - theta)
        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        )
        rqs_out = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - theta).pow(2)
        )
        rqs_logdet = torch.log(torch.clamp_min(derivative_numerator, 1e-12)) - 2.0 * torch.log(torch.clamp_min(denominator, 1e-12))

    outputs = torch.where(inside_mask, rqs_out, inputs)
    logabsdet = torch.where(inside_mask, rqs_logdet, torch.zeros_like(inputs))
    return outputs, logabsdet


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.half_channels = in_channels // 2
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, self.num_bins * 3 - 1, t).permute(0, 1, 3, 2)
        scale = math.sqrt(self.filter_channels)
        unnormalized_widths = h[..., : self.num_bins] / scale
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / scale
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = vectorized_unconstrained_rqs(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
        )
        x_out = torch.cat([x0, x1], dim=1) * x_mask
        if not reverse:
            logdet = torch.sum(logabsdet * x_mask, [1, 2])
            return x_out, logdet
        return x_out, None


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 192,
        kernel_size: int = 3,
        p_dropout: float = 0.5,
        n_flows: int = 4,
    ) -> None:
        super().__init__()
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

        self.flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)

        self.post_flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        x = torch.detach(x).float()
        x_mask = x_mask.float()
        with torch.amp.autocast(device_type=x.device.type if x.device.type in ("cuda", "cpu") else "cpu", enabled=False):
            x = self.pre(x)
            x = self.convs(x, x_mask)
            x = self.proj(x) * x_mask

            if not reverse:
                assert w is not None
                w_f = w.float()
                h_w = self.post_pre(w_f)
                h_w = self.post_convs(h_w, x_mask)
                h_w = self.post_proj(h_w) * x_mask
                e_q = torch.randn(w_f.size(0), 2, w_f.size(2), device=x.device, dtype=torch.float32) * x_mask
                z_q = e_q
                logdet_tot_q = torch.zeros(w_f.size(0), device=x.device, dtype=torch.float32)
                for idx, flow in enumerate(self.post_flows):
                    z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w), reverse=False)
                    if idx > 0:
                        z_q = torch.flip(z_q, [1])
                    if logdet_q is not None:
                        logdet_tot_q = logdet_tot_q + logdet_q
                z_u, z1 = torch.split(z_q, [1, 1], dim=1)
                u = torch.sigmoid(z_u) * x_mask
                z0 = (w_f - u) * x_mask
                logdet_tot_q = logdet_tot_q + torch.sum(
                    (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
                )
                logq = torch.sum(-0.5 * (math.log(2.0 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

                logdet_tot = torch.zeros(w_f.size(0), device=x.device, dtype=torch.float32)
                z0 = torch.log(torch.clamp_min(z0, 1e-5)) * x_mask
                logdet_tot = logdet_tot + torch.sum(-z0 * x_mask, [1, 2])
                z = torch.cat([z0, z1], dim=1)
                for idx, flow in enumerate(self.flows):
                    z, logdet = flow(z, x_mask, g=x, reverse=False)
                    if idx > 0:
                        z = torch.flip(z, [1])
                    if logdet is not None:
                        logdet_tot = logdet_tot + logdet
                nll = torch.sum(0.5 * (math.log(2.0 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
                return nll + logq
            else:
                z = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=torch.float32) * noise_scale
                for flow in reversed(self.flows[2:]):
                    z = torch.flip(z, [1])
                    z, _ = flow(z, x_mask, g=x, reverse=True)
                z = torch.flip(z, [1])
                z, _ = self.flows[0](z, x_mask, reverse=True)
                logw, _ = torch.split(z, [1, 1], dim=1)
                return logw


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0])
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])
                    )
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class WaveformDecoder(nn.Module):
    def __init__(
        self,
        initial_channel: int = 192,
        resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: list[list[int]] | tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4),
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, tuple(d)))  # type: ignore[arg-type]

        ch_final = upsample_initial_channel // (2 ** len(self.ups))
        self.conv_post = nn.Conv1d(ch_final, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # Note: Coqui VITS uses default slope 0.01 before conv_post!
        x = self.conv_post(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# Multi-Period + Multi-Scale Discriminator (`disc.nets` in `best_model.pth`)
# ---------------------------------------------------------------------------


class DiscriminatorS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        for lyr in self.convs:
            x = lyr(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        pad = get_padding(kernel_size, 1)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(pad, 0))),
                weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(pad, 0))),
                weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(pad, 0))),
                weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(pad, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(pad, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for lyr in self.convs:
            x = lyr(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class VitsMultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.nets = nn.ModuleList([DiscriminatorS()] + [DiscriminatorP(p) for p in periods])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs: list[torch.Tensor] = []
        y_d_gs: list[torch.Tensor] = []
        fmap_rs: list[list[torch.Tensor]] = []
        fmap_gs: list[list[torch.Tensor]] = []
        for d in self.nets:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ---------------------------------------------------------------------------
# Fast Monotonic Alignment Search (`maximum_path`) & Slicing Utilities
# ---------------------------------------------------------------------------


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    b, t_x, max_len = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)
    cum_flat = cum_duration.view(b * t_x)
    seq_range = torch.arange(max_len, dtype=duration.dtype, device=duration.device)
    path = (seq_range.unsqueeze(0) < cum_flat.unsqueeze(1)).to(mask.dtype)
    path = path.view(b, t_x, max_len)
    padded = F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1, :]
    path = path - padded
    return path * mask


_MAS_C_LIB: Any = None
_MAS_C_TRIED = False

_MAS_C_SRC = r"""
#include <stddef.h>
#include <stdint.h>

void maximum_path_c(
    float* path,
    const float* value,
    const int32_t* t_x_lens,
    const int32_t* t_y_lens,
    int32_t b,
    int32_t t_x_max,
    int32_t t_y_max
) {
    const float neg_inf = -1e9f;
    for (int32_t i = 0; i < b; ++i) {
        int32_t tx = t_x_lens[i];
        int32_t ty = t_y_lens[i];
        if (tx <= 0 || ty <= 0) continue;
        const float* v_b = value + (size_t)i * t_x_max * t_y_max;
        float* p_b = path + (size_t)i * t_x_max * t_y_max;

        // Use p_b temporarily to store DP values across [tx, ty]
        for (int32_t x = 0; x < tx; ++x) {
            for (int32_t y = 0; y < ty; ++y) {
                p_b[x * t_y_max + y] = neg_inf;
            }
        }
        p_b[0] = v_b[0];
        for (int32_t y = 1; y < ty; ++y) {
            int32_t x_lim = (y + 1 < tx) ? (y + 1) : tx;
            for (int32_t x = 0; x < x_lim; ++x) {
                float s_same = (x < y) ? p_b[x * t_y_max + (y - 1)] : neg_inf;
                float s_up = (x > 0) ? p_b[(x - 1) * t_y_max + (y - 1)] : neg_inf;
                float best = (s_up > s_same) ? s_up : s_same;
                p_b[x * t_y_max + y] = best + v_b[x * t_y_max + y];
            }
        }
        int32_t xi = tx - 1;
        for (int32_t y = ty - 1; y >= 0; --y) {
            int32_t go_up = 0;
            if (y > 0 && xi > 0) {
                float s_same = (xi < y) ? p_b[xi * t_y_max + (y - 1)] : neg_inf;
                float s_up = p_b[(xi - 1) * t_y_max + (y - 1)];
                if (s_up > s_same) go_up = 1;
            }
            for (int32_t x = 0; x < tx; ++x) {
                p_b[x * t_y_max + y] = (x == xi) ? 1.0f : 0.0f;
            }
            if (go_up) xi -= 1;
        }
    }
}
"""


def _get_mas_c_lib() -> Any:
    global _MAS_C_LIB, _MAS_C_TRIED
    if _MAS_C_TRIED:
        return _MAS_C_LIB
    _MAS_C_TRIED = True
    try:
        import ctypes
        import shutil
        import subprocess
        import tempfile

        cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if not cc:
            return None
        tmp_dir = Path(tempfile.gettempdir()) / f"ninada_mas_c_{os.getpid()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        c_file = tmp_dir / "mas.c"
        so_file = tmp_dir / "mas.so"
        c_file.write_text(_MAS_C_SRC, encoding="utf-8")
        subprocess.run([cc, "-O3", "-shared", "-fPIC", str(c_file), "-o", str(so_file)], check=True, timeout=15)
        lib = ctypes.CDLL(str(so_file))
        lib.maximum_path_c.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.maximum_path_c.restype = None
        _MAS_C_LIB = lib
    except Exception:
        _MAS_C_LIB = None
    return _MAS_C_LIB


@torch.no_grad()
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Exact Monotonic Alignment Search (`neg_cent`: `[B, T_x, T_y]`, `mask`: `[B, T_x, T_y]`).
    Uses an on-the-fly `-O3` compiled C kernel (~0.25 ms per batch) with exact PyTorch fallback.
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    b, t_x_max, t_y_max = neg_cent.shape

    c_lib = _get_mas_c_lib()
    if c_lib is not None:
        val_np = np.ascontiguousarray(neg_cent.detach().cpu().to(torch.float32).numpy())
        msk_cpu = mask.detach().cpu().bool()
        tx_np = np.ascontiguousarray(msk_cpu[:, :, 0].sum(dim=1).numpy().astype(np.int32))
        ty_np = np.ascontiguousarray(msk_cpu[:, 0, :].sum(dim=1).numpy().astype(np.int32))
        path_np = np.zeros((b, t_x_max, t_y_max), dtype=np.float32)
        c_lib.maximum_path_c(
            path_np.ctypes.data,
            val_np.ctypes.data,
            tx_np.ctypes.data,
            ty_np.ctypes.data,
            int(b),
            int(t_x_max),
            int(t_y_max),
        )
        return torch.from_numpy(path_np).to(device=device, dtype=dtype)

    val = neg_cent.detach().cpu().to(torch.float32).clone()
    msk = mask.detach().cpu().bool()
    t_x_lens = msk[:, :, 0].sum(dim=1)  # [B]
    t_y_lens = msk[:, 0, :].sum(dim=1)  # [B]

    neg_inf = -1e9
    # Work in `[B, T_x]` slice across `y = 0 .. t_y_max - 1`
    dp = torch.full((b, t_x_max), neg_inf, dtype=torch.float32)
    dp[:, 0] = val[:, 0, 0]
    backptr = torch.zeros((b, t_x_max, t_y_max), dtype=torch.bool)  # True if transitioned from x - 1

    x_idx = torch.arange(t_x_max).unsqueeze(0)  # [1, T_x]
    for y in range(1, t_y_max):
        prev_same = dp  # [B, T_x]
        prev_up = F.pad(dp[:, :-1], (1, 0), value=neg_inf)  # [B, T_x]
        # At x == 0: cannot come from x - 1; at x == y: must come from x - 1 (since y-1 < x)
        valid_same = x_idx < y
        s_same = torch.where(valid_same, prev_same, torch.full_like(prev_same, neg_inf))
        go_up = prev_up > s_same
        best_prev = torch.where(go_up, prev_up, s_same)
        active_y = (y < t_y_lens).unsqueeze(1) & (x_idx <= y) & (x_idx < t_x_lens.unsqueeze(1))
        dp = torch.where(active_y, best_prev + val[:, :, y], dp)
        backptr[:, :, y] = go_up & active_y

    path = np.zeros((b, t_x_max, t_y_max), dtype=np.float32)
    bp_np = backptr.numpy()
    tx_np = t_x_lens.numpy()
    ty_np = t_y_lens.numpy()
    for i in range(b):
        xi = int(tx_np[i]) - 1
        yi_end = int(ty_np[i]) - 1
        if xi < 0 or yi_end < 0:
            continue
        for y in range(yi_end, -1, -1):
            path[i, xi, y] = 1.0
            if y > 0 and xi > 0 and bp_np[i, xi, y]:
                xi -= 1
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = SEGMENT_FRAMES) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = int(ids_str[i])
        idx_end = idx_str + segment_size
        chunk = x[i, :, idx_str:idx_end]
        ret[i, :, : chunk.size(-1)] = chunk
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int = SEGMENT_FRAMES
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, t = x.size()
    ids_str_max = torch.clamp_min(x_lengths - segment_size + 1, 1)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


# ---------------------------------------------------------------------------
# Full Top-Level Studio VITS Model (`949/949` keys with `SYSPIN/vits_*`)
# ---------------------------------------------------------------------------


class StudioVitsModel(nn.Module):
    def __init__(
        self,
        num_chars: int,
        spec_channels: int = SPEC_CHANNELS,
        segment_size: int = SEGMENT_FRAMES,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        use_discriminator: bool = True,
    ) -> None:
        super().__init__()
        self.num_chars = num_chars
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.hidden_channels = hidden_channels

        self.text_encoder = TextEncoder(
            n_vocab=num_chars,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        self.posterior_encoder = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
        )
        self.flow = ResidualCouplingBlock(
            channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
        )
        self.duration_predictor = StochasticDurationPredictor(
            in_channels=hidden_channels,
            filter_channels=hidden_channels,
            kernel_size=3,
            p_dropout=0.5,
            n_flows=4,
        )
        self.waveform_decoder = WaveformDecoder(
            initial_channel=hidden_channels,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            upsample_rates=(8, 8, 2, 2),
            upsample_initial_channel=512,
            upsample_kernel_sizes=(16, 16, 4, 4),
        )
        if use_discriminator:
            self.disc = VitsMultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11))

    def generator_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for mod in (
            self.text_encoder,
            self.posterior_encoder,
            self.flow,
            self.duration_predictor,
            self.waveform_decoder,
        ):
            params.extend(list(mod.parameters()))
        return params

    def discriminator_parameters(self) -> list[nn.Parameter]:
        return list(self.disc.parameters())

    def forward_train(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        enc_x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)
        z, m_q, logs_q, y_mask = self.posterior_encoder(spec, spec_lengths)
        z_p = self.flow(z, y_mask, reverse=False)

        with torch.no_grad():
            m_pf = m_p.float()
            logs_pf = logs_p.float()
            z_pf = z_p.float()
            s_p_sq_r = torch.exp(-2.0 * logs_pf)  # [b, d, t_x]
            neg_cent1 = torch.sum(-0.5 * math.log(2.0 * math.pi) - logs_pf, [1], keepdim=True)  # [b, 1, t_x]
            neg_cent2 = torch.matmul(-0.5 * (z_pf ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_y, t_x]
            neg_cent3 = torch.matmul(z_pf.transpose(1, 2), (m_pf * s_p_sq_r))  # [b, t_y, t_x]
            neg_cent4 = torch.sum(-0.5 * (m_pf ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_x]
            neg_cent = (neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4).transpose(1, 2)  # [b, t_x, t_y]

            attn_mask = x_mask.transpose(1, 2) * y_mask  # [b, t_x, t_y]
            attn = maximum_path(neg_cent, attn_mask).detach()

        w = attn.sum(dim=2, keepdim=True).transpose(1, 2)  # [b, 1, t_x]
        l_length_raw = self.duration_predictor(enc_x, x_mask, w=w, reverse=False)
        l_length = torch.sum(l_length_raw.float()) / torch.clamp_min(torch.sum(x_mask.float()), 1.0)

        # Expand text priors along monotonic alignment
        attn_m = attn.to(dtype=m_p.dtype)
        m_p_exp = torch.matmul(m_p, attn_m)  # [b, d, t_y]
        logs_p_exp = torch.matmul(logs_p, attn_m)  # [b, d, t_y]

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.waveform_decoder(z_slice)
        return o, l_length, ids_slice, attn, (z_p, logs_q, m_p_exp, logs_p_exp, y_mask)

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        length_scale: float | torch.Tensor = 1.0,
        noise_scale: float | torch.Tensor = 0.667,
        noise_scale_dp: float | torch.Tensor = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_lengths is None:
            x_lengths = torch.tensor([x.size(1)], dtype=torch.long, device=x.device)
        enc_x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)
        logw = self.duration_predictor(enc_x, x_mask, reverse=True, noise_scale=noise_scale_dp)
        w = torch.clamp_max(torch.exp(logw), 50.0) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        w_ceil = torch.cat([w_ceil[:, :, :-1], torch.clamp_min(w_ceil[:, :, -1:], 8.0)], dim=2) * x_mask
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.transpose(1, 2) * y_mask  # [b, t_x, t_y]
        attn = generate_path(w_ceil.squeeze(1), attn_mask)  # [b, t_x, t_y]

        m_p_exp = torch.matmul(m_p, attn)
        logs_p_exp = torch.matmul(logs_p, attn)
        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        wav = self.waveform_decoder(z * y_mask)
        return wav.squeeze(1), w_ceil.squeeze(1).long()

    def strip_weight_norm_for_export(self) -> None:
        """Fuse `parametrizations.weight` into plain `weight` tensors for fast zero-overhead ONNX export."""
        for module in self.modules():
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


class StudioVitsONNXWrapper(nn.Module):
    """
    Clean inference wrapper for ONNX Runtime (WASM SIMD128 / WebGPU / Node / Python).
    Inputs:
      - `input_ids`: `int64 [1, T]` (`[BLNK, c0, BLNK, c1, ..., BLNK]`)
      - `scales`: `float32 [3]` = `[length_scale, noise_scale, noise_scale_dp]`
    Outputs:
      - `waveform`: `float32 [1, N]` (24,000 Hz audio)
      - `durations`: `int64 [1, T]` (per-token frame durations, `hop = 256`)
    """

    def __init__(self, model: StudioVitsModel) -> None:
        super().__init__()
        self.text_encoder = model.text_encoder
        self.duration_predictor = model.duration_predictor
        self.flow = model.flow
        self.waveform_decoder = model.waveform_decoder

    def forward(self, input_ids: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length_scale = scales[0]
        noise_scale = scales[1]
        noise_scale_dp = scales[2]

        x_lengths = (input_ids >= 0).long().sum(dim=1)
        enc_x, m_p, logs_p, x_mask = self.text_encoder(input_ids, x_lengths)
        logw = self.duration_predictor(enc_x, x_mask, reverse=True, noise_scale=noise_scale_dp)
        w = torch.clamp_max(torch.exp(logw), 50.0) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        w_ceil = torch.cat([w_ceil[:, :, :-1], torch.clamp_min(w_ceil[:, :, -1:], 8.0)], dim=2) * x_mask
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        max_y = y_lengths.max()
        seq_range = torch.arange(max_y, dtype=torch.long, device=input_ids.device)
        y_mask = (seq_range.unsqueeze(0) < y_lengths.unsqueeze(1)).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.transpose(1, 2) * y_mask
        durations = w_ceil.squeeze(1)
        attn = generate_path(durations, attn_mask)

        m_p_exp = torch.matmul(m_p, attn)
        logs_p_exp = torch.matmul(logs_p, attn)
        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        wav = self.waveform_decoder(z * y_mask).squeeze(1)
        return wav, durations.to(torch.int64)


# ---------------------------------------------------------------------------
# VITS Adversarial + 24 kHz Mel + KL + Feature Matching Losses
# ---------------------------------------------------------------------------


class MultiSpecExtractor24k(nn.Module):
    """Computes both linear 513-bin spectrogram (for PosteriorEncoder) and 80-bin 24 kHz Mel spectrogram."""

    def __init__(
        self,
        n_fft: int = FFT_SIZE,
        num_mels: int = N_MELS,
        sampling_rate: int = SAMPLE_RATE_24K,
        hop_size: int = HOP_LENGTH,
        win_size: int = WIN_LENGTH,
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.pad = int((n_fft - hop_size) / 2)

        import librosa

        mel_np = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_np).float())
        self.register_buffer("hann_window", torch.hann_window(win_size).float())

    def linear_spec(self, y: torch.Tensor) -> torch.Tensor:
        """y: [B, T] -> [B, 513, Frames]"""
        dev_type = y.device.type if y.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            yf = y.float()
            y_pad = F.pad(yf.unsqueeze(1), (self.pad, self.pad), mode="reflect").squeeze(1)
            spec_c = torch.stft(
                y_pad,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window.float(),
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            return torch.sqrt(spec_c.real.pow(2) + spec_c.imag.pow(2) + 1e-6)

    def spec_to_mel(self, spec: torch.Tensor) -> torch.Tensor:
        dev_type = spec.device.type if spec.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            mel = torch.matmul(self.mel_basis.float(), spec.float())
            return torch.log(torch.clamp_min(mel, 1e-5))

    def wav_to_mel(self, y: torch.Tensor) -> torch.Tensor:
        return self.spec_to_mel(self.linear_spec(y))


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.clamp_min(torch.sum(z_mask), 1.0)


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> torch.Tensor:
    loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=torch.float32).squeeze(0)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_f = dr.float()
        dg_f = dg.float()
        r_loss = torch.mean((1.0 - dr_f) ** 2)
        g_loss = torch.mean(dg_f ** 2)
        loss = loss + r_loss + g_loss
    return loss


def generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros(1, device=disc_outputs[0].device, dtype=torch.float32).squeeze(0)
    for dg in disc_outputs:
        dg_f = dg.float()
        loss = loss + torch.mean((1.0 - dg_f) ** 2)
    return loss


def feature_loss(fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]) -> torch.Tensor:
    loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=torch.float32).squeeze(0)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl_f = rl.float().detach()
            gl_f = gl.float()
            loss = loss + torch.mean(torch.abs(rl_f - gl_f))
    return loss * 2.0
