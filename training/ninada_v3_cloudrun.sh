#!/usr/bin/env bash
set -euo pipefail

echo "=== [Ninada v3.0 24 kHz GCP Studio Trainer] Booting on $(hostname) ==="
nvidia-smi

pip install -q google-cloud-storage google-crc32c pyarrow soundfile librosa scipy onnx onnxruntime safetensors huggingface_hub

mkdir -p /tmp/ninada_v3_work /tmp/ninada_v3_data
cd /tmp/ninada_v3_work

python3 -c "
import os
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('training-gemina')
bucket.blob('ninada_v3/ninada_v3_bundle.tar.gz').download_to_filename('/tmp/ninada_v3_bundle.tar.gz')
tok_blob = bucket.blob('ninada_v3/.hf_token')
if tok_blob.exists():
    tok_blob.download_to_filename('/tmp/.hf_token')
cache_blob = bucket.blob('ninada_v3/curated_24k_cache.tar.gz')
if cache_blob.exists():
    print('Found cached curated_24k_cache.tar.gz in GCS — downloading...')
    cache_blob.download_to_filename('/tmp/curated_24k_cache.tar.gz')
print('Downloaded gs://training-gemina/ninada_v3/ninada_v3_bundle.tar.gz')
"

if [ -f /tmp/.hf_token ]; then
  export HF_TOKEN="$(tr -d '\r\n' < /tmp/.hf_token)"
  rm -f /tmp/.hf_token
fi

tar -xzf /tmp/ninada_v3_bundle.tar.gz -C /tmp/ninada_v3_work
rm -f /tmp/ninada_v3_bundle.tar.gz

export PYTHONPATH="/tmp/ninada_v3_work/python/src:/tmp/ninada_v3_work"
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ -f /tmp/curated_24k_cache.tar.gz ]; then
  tar -xzf /tmp/curated_24k_cache.tar.gz -C /tmp/ninada_v3_data
  rm -f /tmp/curated_24k_cache.tar.gz
  echo "Restored curated 24 kHz studio dataset from GCS cache!"
else
  python3 training/train_gcp_24k.py \
    --work-dir /tmp/ninada_v3_data \
    --out-dir /tmp/ninada_v3_work/artifacts/ninada_v3 \
    --per-lang-target 400 \
    --curate-only

  tar -czf /tmp/curated_24k_cache.tar.gz -C /tmp/ninada_v3_data .
  python3 -c "
import os
from google.cloud import storage
sz = os.path.getsize('/tmp/curated_24k_cache.tar.gz') / (1024 * 1024)
print(f'Caching curated_24k_cache.tar.gz ({sz:.2f} MB) to GCS...', flush=True)
client = storage.Client()
bucket = client.bucket('training-gemina')
bucket.blob('ninada_v3/curated_24k_cache.tar.gz').upload_from_filename('/tmp/curated_24k_cache.tar.gz', timeout=900)
print('Cached curated_24k_cache.tar.gz successfully!')
"
  rm -f /tmp/curated_24k_cache.tar.gz
fi

python3 training/train_gcp_24k.py \
  --work-dir /tmp/ninada_v3_data \
  --out-dir /tmp/ninada_v3_work/artifacts/ninada_v3 \
  --epochs 2 \
  --per-lang-target 400 \
  --grad-accum 4

# Copy held-out split manifests for local scientific bootstrap benchmarking!
cp /tmp/ninada_v3_data/val_v3.jsonl /tmp/ninada_v3_work/artifacts/ninada_v3/ || true
cp /tmp/ninada_v3_data/test_v3.jsonl /tmp/ninada_v3_work/artifacts/ninada_v3/ || true

# Bundle the 240 held-out test wavs (~65 MB total) so local objective + ear benchmarks compare against exact studio ground truth!
mkdir -p /tmp/ninada_v3_work/artifacts/ninada_v3/test_gt_wavs
python3 -c "
import json, shutil
from pathlib import Path
dst = Path('/tmp/ninada_v3_work/artifacts/ninada_v3/test_gt_wavs')
for line in open('/tmp/ninada_v3_data/test_v3.jsonl', encoding='utf-8'):
    rec = json.loads(line)
    p = Path(rec['wav_path'])
    if p.exists():
        shutil.copy2(p, dst / p.name)
print('Bundled held-out test ground-truth wavs:', len(list(dst.glob('*.wav'))))
"

rm -rf /tmp/ninada_v3_data ~/.cache/huggingface /root/.cache/huggingface || true

tar -czf /tmp/ninada_v3_artifacts.tar.gz -C /tmp/ninada_v3_work/artifacts ninada_v3

python3 -c "
import os
from google.cloud import storage
sz = os.path.getsize('/tmp/ninada_v3_artifacts.tar.gz') / (1024 * 1024)
print(f'Uploading ninada_v3_artifacts.tar.gz ({sz:.2f} MB)...', flush=True)
client = storage.Client()
bucket = client.bucket('training-gemina')
blob = bucket.blob('ninada_v3/ninada_v3_artifacts.tar.gz')
blob.upload_from_filename('/tmp/ninada_v3_artifacts.tar.gz', timeout=1200)
print('Uploaded gs://training-gemina/ninada_v3/ninada_v3_artifacts.tar.gz successfully!')
"

echo "=== [Ninada v3.0 24 kHz GCP Studio Trainer] ALL DONE ==="
