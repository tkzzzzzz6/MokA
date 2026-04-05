#!/usr/bin/env bash
set -euo pipefail

# Must run from AudioVisualText root for relative paths to match training code.
if [[ ! -f "scripts/pretrain/pretrain_audio.sh" ]]; then
  echo "[error] Please run from AudioVisualText root directory."
  exit 1
fi

command -v python3 >/dev/null 2>&1 || {
  echo "[error] python3 not found"
  exit 1
}

# Increase huggingface_hub timeouts for unstable networks.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# If huggingface.co resolves to loopback in WSL, route through mirror endpoint.
hf_ip="$(python3 - <<'PY'
import socket
try:
    print(socket.gethostbyname('huggingface.co'))
except Exception:
    print('')
PY
)"

HF_ENDPOINT_ARG=""
if [[ "$hf_ip" == 127.* ]]; then
  echo "[warn] huggingface.co resolves to $hf_ip, enabling HF mirror endpoint."
  HF_ENDPOINT_ARG="--hf_endpoint https://hf-mirror.com"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[info] ffmpeg not found, installing via apt..."
  sudo apt-get update
  sudo apt-get install -y ffmpeg
fi

python3 -m pip install -U datasets yt-dlp

python3 scripts/pretrain/prepare_audiocaps.py \
  --dataset d0rj/audiocaps \
  --train_n 500 \
  --val_n 100 \
  --out_dir AudioCaps \
  --retry 2 \
  --hf_timeout 60 \
  ${HF_ENDPOINT_ARG}

echo "[ok] AudioCaps subset ready under AudioCaps/"
echo "[next] Run: bash scripts/pretrain/pretrain_audio.sh"
