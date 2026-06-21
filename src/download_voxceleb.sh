#!/usr/bin/env bash
set -euo pipefail

# Downloads VoxCeleb1 (dev + test audio + metadata) from the gated
# ProgramComputer/voxceleb dataset on Hugging Face.
# Requires HF_TOKEN in a .env file (accept the dataset terms first).

source .env

DATA_DIR="data"
VOX_DIR="${DATA_DIR}/vox_celeb12"
BASE="https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1"
AUTH="Authorization: Bearer ${HF_TOKEN}"

mkdir -p "${VOX_DIR}"

wget --header="${AUTH}" -O "${VOX_DIR}/vox1_meta.csv" "${BASE}/vox1_meta.csv"

for name in vox1_dev_wav vox1_test_wav vox1_dev_txt vox1_test_txt; do
  wget --header="${AUTH}" -O "${VOX_DIR}/${name}.zip" "${BASE}/${name}.zip"
  unzip -q "${VOX_DIR}/${name}.zip" -d "${VOX_DIR}/${name}"
  rm "${VOX_DIR}/${name}.zip"
done

