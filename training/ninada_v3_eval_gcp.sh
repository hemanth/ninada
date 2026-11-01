#!/usr/bin/env bash
set -euo pipefail

echo "=== [Ninada v3.0 24 kHz Cloud GPU Eval Optimizer & Benchmark] Booting on $(hostname) ==="
nvidia-smi

pip install -q google-cloud-storage google-crc32c pyarrow soundfile librosa scipy onnx onnxruntime safetensors huggingface_hub

mkdir -p /tmp/ninada_v3_work /tmp/ninada_v3_data /tmp/ninada_v3_raw
cd /tmp/ninada_v3_work

python3 -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('training-gemina')
bucket.blob('ninada_v3/ninada_v3_eval_bundle.tar.gz').download_to_filename('/tmp/ninada_v3_eval_bundle.tar.gz')
bucket.blob('ninada_v3/curated_24k_cache.tar.gz').download_to_filename('/tmp/curated_24k_cache.tar.gz')
bucket.blob('ninada_v3/ninada_v3_artifacts.tar.gz').download_to_filename('/tmp/ninada_v3_artifacts.tar.gz')
tok_blob = bucket.blob('ninada_v3/.hf_token')
if tok_blob.exists():
    tok_blob.download_to_filename('/tmp/.hf_token')
print('Downloaded eval bundle + curated 24k studio cache + raw trained checkpoint from GCS!')
"

if [ -f /tmp/.hf_token ]; then
  export HF_TOKEN="$(tr -d '\r\n' < /tmp/.hf_token)"
  rm -f /tmp/.hf_token
fi

tar -xzf /tmp/ninada_v3_eval_bundle.tar.gz -C /tmp/ninada_v3_work
tar -xzf /tmp/curated_24k_cache.tar.gz -C /tmp/ninada_v3_data
tar -xzf /tmp/ninada_v3_artifacts.tar.gz -C /tmp/ninada_v3_raw
rm -f /tmp/ninada_v3_eval_bundle.tar.gz /tmp/curated_24k_cache.tar.gz /tmp/ninada_v3_artifacts.tar.gz

export PYTHONPATH="/tmp/ninada_v3_work/python/src:/tmp/ninada_v3_work"
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export NINADA_ORT_THREADS=6
export OMP_NUM_THREADS=4

mkdir -p /tmp/ninada_v3_work/artifacts/ninada_v3
cp /tmp/ninada_v3_raw/ninada_v3/val_test_metrics_v3.json /tmp/ninada_v3_work/artifacts/ninada_v3/ || true
cp /tmp/ninada_v3_data/val_v3.jsonl /tmp/ninada_v3_work/artifacts/ninada_v3/
cp /tmp/ninada_v3_data/test_v3.jsonl /tmp/ninada_v3_work/artifacts/ninada_v3/
cp -r /tmp/ninada_v3_raw/ninada_v3/test_gt_wavs /tmp/ninada_v3_work/artifacts/ninada_v3/

# Step 1: Leakage-free GPU calibration on val_v3.jsonl (N=240) + ONNX & voice bank export
python3 training/optimize_eval_gcp.py \
  --raw-ckpt /tmp/ninada_v3_raw/ninada_v3/model.safetensors \
  --val-jsonl /tmp/ninada_v3_data/val_v3.jsonl \
  --out-dir /tmp/ninada_v3_work/artifacts/ninada_v3

# Step 2: Full 6-system N=240 held-out studio test evaluation (GPU PyTorch + CPU ONNX Runtime) + threshold gate
python3 bench/run_bench.py \
  --artifact-dir /tmp/ninada_v3_work/artifacts/ninada_v3 \
  --out-json /tmp/ninada_v3_work/bench/results.json \
  --out-md /tmp/ninada_v3_work/bench/SCIENCE_REPORT.md

python3 bench/check_bench_thresholds.py

# Step 3: Package calibrated artifacts + benchmark outputs back to GCS
mkdir -p /tmp/ninada_v3_eval_out/artifacts/ninada_v3 /tmp/ninada_v3_eval_out/bench
cp /tmp/ninada_v3_work/bench/results.json /tmp/ninada_v3_eval_out/bench/
cp /tmp/ninada_v3_work/bench/SCIENCE_REPORT.md /tmp/ninada_v3_eval_out/bench/
cp /tmp/ninada_v3_work/artifacts/ninada_v3/model.safetensors /tmp/ninada_v3_eval_out/artifacts/ninada_v3/
cp /tmp/ninada_v3_work/artifacts/ninada_v3/config_v3.json /tmp/ninada_v3_eval_out/artifacts/ninada_v3/
cp /tmp/ninada_v3_work/artifacts/ninada_v3/val_calibration_report.json /tmp/ninada_v3_eval_out/artifacts/ninada_v3/
cp -r /tmp/ninada_v3_work/artifacts/ninada_v3/onnx /tmp/ninada_v3_eval_out/artifacts/ninada_v3/
cp -r /tmp/ninada_v3_work/artifacts/ninada_v3/voices /tmp/ninada_v3_eval_out/artifacts/ninada_v3/
cp -r /tmp/ninada_v3_work/artifacts/ninada_v3/samples_v3 /tmp/ninada_v3_eval_out/artifacts/ninada_v3/

tar -czf /tmp/ninada_v3_eval_outputs.tar.gz -C /tmp/ninada_v3_eval_out .

python3 -c "
import os
from google.cloud import storage
sz = os.path.getsize('/tmp/ninada_v3_eval_outputs.tar.gz') / (1024 * 1024)
print(f'Uploading ninada_v3_eval_outputs.tar.gz ({sz:.2f} MB)...', flush=True)
client = storage.Client()
bucket = client.bucket('training-gemina')
bucket.blob('ninada_v3/ninada_v3_eval_outputs.tar.gz').upload_from_filename('/tmp/ninada_v3_eval_outputs.tar.gz', timeout=900)
print('Uploaded gs://training-gemina/ninada_v3/ninada_v3_eval_outputs.tar.gz successfully!')
"

echo "=== [Ninada v3.0 24 kHz Cloud GPU Eval Optimizer & Benchmark] ALL DONE ==="
