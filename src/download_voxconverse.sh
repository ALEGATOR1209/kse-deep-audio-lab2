#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
AUDIO_DIR="${DATA_DIR}/audio"
ANNOTATIONS_DIR="${DATA_DIR}/annotations"

REPO_URL="https://github.com/joonson/voxconverse.git"
AUDIO_DEV_URL="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
AUDIO_TEST_URL="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
REPO_DIR="${DATA_DIR}/voxconverse"

mkdir -p "${DATA_DIR}"
mkdir -p "${AUDIO_DIR}"
mkdir -p "${ANNOTATIONS_DIR}"

if [ -d "${ANNOTATIONS_DIR}/dev" ] && [ -d "${ANNOTATIONS_DIR}/test" ]; then
  echo "Annotations already present, skipping clone."
else
  git clone "${REPO_URL}" "${REPO_DIR}"
  trap 'rm -rf "${REPO_DIR}"' EXIT
  mv "${REPO_DIR}/dev" "${ANNOTATIONS_DIR}/dev"
  mv "${REPO_DIR}/test" "${ANNOTATIONS_DIR}/test"

  rm -rf "${REPO_DIR}"
fi

if [ -d "${AUDIO_DIR}/dev" ] && [ -d "${AUDIO_DIR}/test" ]; then
  echo "Audio already present, skipping download."
else
  if [ -d "${AUDIO_DIR}/dev" ] && [ -d "${AUDIO_DIR}/test" ]; then
    echo "Audio already downloaded, unpacking."
  else
    wget -O "${DATA_DIR}/voxconverse_dev_wav.zip" "${AUDIO_DEV_URL}"
    wget -O "${DATA_DIR}/voxconverse_test_wav.zip" "${AUDIO_TEST_URL}"
  fi

  unzip "${DATA_DIR}/voxconverse_dev_wav.zip" -d "${AUDIO_DIR}/dev"
  unzip "${DATA_DIR}/voxconverse_test_wav.zip" -d "${AUDIO_DIR}/test"

  mv "${AUDIO_DIR}/dev/audio/"* "${AUDIO_DIR}/dev/"
  rmdir "${AUDIO_DIR}/dev/audio"

  mv "${AUDIO_DIR}/test/voxconverse_test_wav/"* "${AUDIO_DIR}/test/"
  rm -rf "${AUDIO_DIR}/test/__MACOSX"
  rmdir "${AUDIO_DIR}/test/voxconverse_test_wav"

  rm "${DATA_DIR}/voxconverse_dev_wav.zip"
  rm "${DATA_DIR}/voxconverse_test_wav.zip"
fi
