"""Independent verification of benchmark metrics and deployment thresholds for Ninada v5.0 (24,000 Hz Native Per-Language Studio VITS Specialists)."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path("bench/results.json")
    assert results_path.exists(), "bench/results.json not found"
    res = json.loads(results_path.read_text(encoding="utf-8"))

    assert res.get("version") in ("5.0.0", "5.0.1"), f"Expected version 5.0.x, got {res.get('version')}"
    assert res.get("sample_rate_hz") == 24000, f"Expected 24000 Hz, got {res.get('sample_rate_hz')}"
    assert res.get("num_test_utterances") == 480, f"Expected 480 test utterances, got {res.get('num_test_utterances')}"

    sys_pt = res["systems"]["ninada_v3_pytorch"]
    sys_ox = res["systems"]["ninada_v3_onnx"]
    sys_ws = res["systems"]["ninada_v3_warmstart"]
    sys_raw = res["systems"]["mms_hin_raw"]
    sys_flt = res["systems"]["mms_5_fleet"]

    # R1: 100% 6-language coverage & 24,000 Hz output
    assert sys_pt["voiced_rate_pct"] == 100.0, f"FP32 voiced rate {sys_pt['voiced_rate_pct']}% < 100%"
    assert sys_ox["voiced_rate_pct"] == 100.0, f"ONNX voiced rate {sys_ox['voiced_rate_pct']}% < 100%"
    assert sys_ox["sample_rate_hz"] == 24000

    # R2: 24 kHz Native Studio Acoustic & Prosodic Superiority vs 16 kHz MMS Fleet & Untuned Warmstart
    pt_mcd = float(sys_pt["mcd_db"]["mean"])
    ox_mcd = float(sys_ox["mcd_db"]["mean"])
    ws_mcd = float(sys_ws["mcd_db"]["mean"])
    flt_mcd = float(sys_flt["mcd_db"]["mean"])
    raw_mcd = float(sys_raw["mcd_db"]["mean"])

    pt_hf = float(sys_pt["hf_8_12k_err_db"]["mean"])
    ox_hf = float(sys_ox["hf_8_12k_err_db"]["mean"])
    ws_hf = float(sys_ws["hf_8_12k_err_db"]["mean"])
    flt_hf = float(sys_flt["hf_8_12k_err_db"]["mean"])

    pt_dur = float(sys_pt["dur_mae_sec"]["mean"])
    ox_dur = float(sys_ox["dur_mae_sec"]["mean"])
    ws_dur = float(sys_ws["dur_mae_sec"]["mean"])
    flt_dur = float(sys_flt["dur_mae_sec"]["mean"])

    pt_f0 = float(sys_pt["f0_rmse_cents"]["mean"])
    raw_f0 = float(sys_raw["f0_rmse_cents"]["mean"])

    print(
        f"24 kHz MCD (0-8 kHz): Raw={raw_mcd:.2f} dB, MMS-5-Fleet={flt_mcd:.2f} dB, "
        f"Warmstart={ws_mcd:.2f} dB, FP32={pt_mcd:.2f} dB, ONNX={ox_mcd:.2f} dB"
    )
    assert pt_mcd < flt_mcd, f"FP32 MCD {pt_mcd:.2f} did not beat 694 MB MMS fleet {flt_mcd:.2f}"
    assert ox_mcd < flt_mcd, f"ONNX MCD {ox_mcd:.2f} did not beat 694 MB MMS fleet {flt_mcd:.2f}"
    assert pt_mcd < ws_mcd, f"FP32 MCD {pt_mcd:.2f} did not improve over warmstart {ws_mcd:.2f}"
    assert ox_mcd < ws_mcd, f"ONNX MCD {ox_mcd:.2f} did not improve over warmstart {ws_mcd:.2f}"

    print(
        f"Studio 8-12 kHz HF Error: MMS-5-Fleet={flt_hf:.2f} dB, Warmstart={ws_hf:.2f} dB, "
        f"FP32={pt_hf:.2f} dB, ONNX={ox_hf:.2f} dB"
    )
    assert pt_hf < ws_hf and ox_hf < ws_hf, "24 kHz HF error did not improve over warmstart"
    assert ox_hf < flt_hf - 40.0, "24 kHz ONNX HF error did not beat 16 kHz Nyquist brickwall"

    print(f"Duration MAE: MMS-5-Fleet={flt_dur:.2f}s, Warmstart={ws_dur:.2f}s, FP32={pt_dur:.2f}s, ONNX={ox_dur:.2f}s")
    assert pt_dur < flt_dur and ox_dur < flt_dur, "Duration MAE did not beat MMS 5-Specialist Fleet"
    assert pt_dur < ws_dur and ox_dur < ws_dur, "Duration MAE did not improve over warmstart"
    assert pt_f0 < raw_f0, f"F0 RMSE {pt_f0:.1f}c did not beat raw baseline {raw_f0:.1f}c"

    # R3: Active per-language Browser ONNX specialist footprint (<= 38 MB), RTF (<= 0.25x CPU), & parity
    size_mb = float(sys_ox["size_mb"])
    rtf_ox = float(sys_ox["rtf_mean"])
    parity_mcd = float(res["onnx_parity"]["pt_vs_onnx_mcd_db"])
    dur_corr = float(res["onnx_parity"]["duration_corr_r"])
    print(
        f"Native Studio W8A32+FP16 ONNX Specialist: {size_mb:.2f} MB (<=38.0 MB), "
        f"RTF={rtf_ox:.4f}x, Parity MCD={parity_mcd:.2f} dB, Dur r={dur_corr:.4f}"
    )
    assert size_mb <= 38.0, f"ONNX specialist size {size_mb:.2f} MB exceeds 38.0 MB"
    assert rtf_ox <= 0.25, f"ONNX CPU RTF {rtf_ox:.4f}x exceeds 0.25x"
    assert parity_mcd <= 3.0, f"PyTorch vs ONNX parity MCD {parity_mcd:.2f} dB exceeds 3.0 dB"
    assert dur_corr >= 0.95, f"Duration correlation {dur_corr:.4f} < 0.95"

    listening_path = Path("bench/LISTENING_EVALUATION.md")
    assert listening_path.exists(), "bench/LISTENING_EVALUATION.md not found"
    print("ALL NINADA v5.0 24KHZ NATIVE STUDIO SPECIALIST BENCHMARK THRESHOLDS SATISFIED!")


if __name__ == "__main__":
    main()
