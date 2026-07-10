#!/usr/bin/env bash
set -euo pipefail

# Robust A6000 checkpoint downloader for face4.
#
# It first tries a plain curl download from the Hugging Face resolve URL. If
# that fails, it falls back to the Python huggingface_hub API inside the MAT
# micromamba environment. It does not use deprecated `huggingface-cli` and does
# not pass the removed `--local-dir-use-symlinks` flag.

ROOT="${1:-/home/interns/Desktop/face4}"
ENV_PREFIX="${FACE_MAMBA_ENV:-/home/interns/Desktop/mat/.micromamba/envs/mat-a6000}"
MICROMAMBA="${FACE_MICROMAMBA:-$HOME/.local/bin/micromamba}"
RUN=("$MICROMAMBA" run -p "$ENV_PREFIX")

REPO_ID="camenduru/show"
REVISION="064a379f415f674051145ec4862f54bd6a65073f"
FILENAME="models/arcface/ms1mv3_arcface_r100_fp16.pth"
URL="https://huggingface.co/${REPO_ID}/resolve/${REVISION}/${FILENAME}?download=true"
EXPECTED_SHA="a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b"

OUT_DIR="$ROOT/models/arcface"
OUT="$OUT_DIR/iresnet100.pth"
TMP="$OUT_DIR/iresnet100.pth.tmp"

mkdir -p "$OUT_DIR"

echo "[face-ckpt] target: $OUT"
echo "[face-ckpt] source: $URL"

if [[ -f "$OUT" ]]; then
  CURRENT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
  if [[ "$CURRENT_SHA" == "$EXPECTED_SHA" ]]; then
    echo "[face-ckpt] existing checkpoint is valid."
    "${RUN[@]}" python -m face4.scripts.setup_arcface \
      --checkpoint-path "$OUT" \
      --source-name "camenduru/show ${FILENAME} revision ${REVISION}"
    exit 0
  fi
  BAD="$OUT.bad.$(date +%Y%m%d_%H%M%S)"
  echo "[face-ckpt] existing checkpoint SHA mismatch:"
  echo "[face-ckpt]   got:      $CURRENT_SHA"
  echo "[face-ckpt]   expected: $EXPECTED_SHA"
  echo "[face-ckpt] moving bad file to: $BAD"
  mv "$OUT" "$BAD"
fi

rm -f "$TMP"

echo "[face-ckpt] trying curl download..."
if command -v curl >/dev/null 2>&1; then
  if curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 -o "$TMP" "$URL"; then
    GOT_SHA="$(sha256sum "$TMP" | awk '{print $1}')"
    if [[ "$GOT_SHA" == "$EXPECTED_SHA" ]]; then
      mv "$TMP" "$OUT"
      echo "[face-ckpt] curl download ok."
    else
      echo "[face-ckpt] curl SHA mismatch:"
      echo "[face-ckpt]   got:      $GOT_SHA"
      echo "[face-ckpt]   expected: $EXPECTED_SHA"
      rm -f "$TMP"
    fi
  else
    echo "[face-ckpt] curl download failed; will try huggingface_hub fallback."
    rm -f "$TMP"
  fi
else
  echo "[face-ckpt] curl not found; will try huggingface_hub fallback."
fi

if [[ ! -f "$OUT" ]]; then
  echo "[face-ckpt] installing/checking huggingface_hub fallback dependencies..."
  "${RUN[@]}" python -m pip install "huggingface_hub[hf_xet]>=0.36,<2" >/tmp/face_hf_install.log 2>&1 || {
    cat /tmp/face_hf_install.log
    echo "[face-ckpt] huggingface_hub install failed."
    exit 1
  }

  echo "[face-ckpt] trying huggingface_hub Python download..."
  FACE_ROOT="$ROOT" FACE_REPO_ID="$REPO_ID" FACE_REVISION="$REVISION" FACE_FILENAME="$FILENAME" FACE_TMP="$TMP" \
    "${RUN[@]}" python - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

root = os.environ["FACE_ROOT"]
path = hf_hub_download(
    repo_id=os.environ["FACE_REPO_ID"],
    filename=os.environ["FACE_FILENAME"],
    revision=os.environ["FACE_REVISION"],
    local_dir=root,
)
shutil.copyfile(path, os.environ["FACE_TMP"])
print(path)
PY

  GOT_SHA="$(sha256sum "$TMP" | awk '{print $1}')"
  if [[ "$GOT_SHA" != "$EXPECTED_SHA" ]]; then
    echo "[face-ckpt] huggingface_hub SHA mismatch:"
    echo "[face-ckpt]   got:      $GOT_SHA"
    echo "[face-ckpt]   expected: $EXPECTED_SHA"
    rm -f "$TMP"
    exit 1
  fi
  mv "$TMP" "$OUT"
  echo "[face-ckpt] huggingface_hub download ok."
fi

echo "[face-ckpt] final SHA:"
sha256sum "$OUT"

echo "[face-ckpt] validating checkpoint with FACE loader..."
"${RUN[@]}" python -m face4.scripts.setup_arcface \
  --checkpoint-path "$OUT" \
  --source-name "camenduru/show ${FILENAME} revision ${REVISION}"

echo "[face-ckpt] done."
