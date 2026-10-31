"""Command-line interface for Ninada (Pan-Indic & Classical Sanskrit TTS)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Ninada


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ninada",
        description="Ninada — Pan-Indic & Classical Sanskrit Neural TTS (16 kHz WAV)",
    )
    parser.add_argument("text", nargs="?", help="Input text in Devanagari, Kannada, Tamil, Telugu, Malayalam, or IAST")
    parser.add_argument("-l", "--lang", default=None, help="Language code: sa | hi | kn | ta | te | ml")
    parser.add_argument("-o", "--output", type=Path, default=Path("ninada_out.wav"), help="Output WAV file path")
    parser.add_argument("-r", "--rate", type=float, default=1.0, help="Speaking rate multiplier (default: 1.0)")
    parser.add_argument("--tier", choices=["auto", "neural", "klatt"], default="auto", help="Synthesis engine tier")
    parser.add_argument("--inspect", action="store_true", help="Print Pan-Indic G2P & akshara prosody JSON")
    args = parser.parse_args()

    txt = args.text if args.text else sys.stdin.read().strip()
    if not txt:
        parser.error("Provide input text as an argument or via stdin.")

    engine = Ninada(lang=args.lang or "sa")
    pcm, meta = engine.synthesize(txt, lang=args.lang, rate=args.rate, tier=args.tier)
    if args.inspect:
        print(json.dumps({k: v for k, v in meta.items() if k != "durations"}, ensure_ascii=False, indent=2))
        return

    from .engine import encode_wav_bytes

    wav = encode_wav_bytes(pcm)
    args.output.write_bytes(wav)
    print(f"Wrote {len(wav)} bytes ({len(pcm)/16000.0:.2f}s, tier={meta['tier']}, lang={meta['lang']}) -> {args.output}")


if __name__ == "__main__":
    main()
