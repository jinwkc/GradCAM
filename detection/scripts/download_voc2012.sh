#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ARCHIVE="$DATA_DIR/VOCtrainval_11-May-2012.tar"
DATASET_DIR="$DATA_DIR/VOCdevkit/VOC2012"
URL="https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

if [[ -d "$DATASET_DIR/JPEGImages" && -d "$DATASET_DIR/Annotations" ]]; then
  echo "PASCAL VOC 2012 train/validation data already exists: $DATASET_DIR"
  exit 0
fi

if [[ -e "$DATA_DIR/VOCdevkit" ]]; then
  echo "Error: $DATA_DIR/VOCdevkit exists but the expected VOC2012 folders are incomplete." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
if ! curl -fL -C - "$URL" -o "$ARCHIVE"; then
  if ! tar -tf "$ARCHIVE" >/dev/null 2>&1; then
    echo "Error: download is incomplete. Run this script again to resume it." >&2
    exit 1
  fi
  echo "The existing archive is complete; continuing with extraction."
fi

STAGING_DIR="$DATA_DIR/.voc2012-extract-$$"
mkdir -p "$STAGING_DIR"
trap 'rm -rf -- "$STAGING_DIR"' EXIT

tar -xf "$ARCHIVE" -C "$STAGING_DIR"
if [[ ! -d "$STAGING_DIR/VOCdevkit/VOC2012/JPEGImages" || \
      ! -d "$STAGING_DIR/VOCdevkit/VOC2012/Annotations" ]]; then
  echo "Error: downloaded archive does not contain the expected VOC2012 folders." >&2
  exit 1
fi

mv "$STAGING_DIR/VOCdevkit" "$DATA_DIR/VOCdevkit"
echo "PASCAL VOC 2012 train/validation data is ready: $DATASET_DIR"
