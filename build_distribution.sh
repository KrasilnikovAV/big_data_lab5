#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="big_data_lab5_openfoodfacts_kmeans"
DIST_DIR="dist"
STAGE_DIR="${DIST_DIR}/${PROJECT_NAME}"
ZIP_PATH="${DIST_DIR}/${PROJECT_NAME}.zip"

rm -rf "${STAGE_DIR}" "${ZIP_PATH}"
mkdir -p "${STAGE_DIR}"

cp README.md "${STAGE_DIR}/"
cp report.md "${STAGE_DIR}/"
cp requirements.txt "${STAGE_DIR}/"
cp Dockerfile "${STAGE_DIR}/"
cp .dockerignore "${STAGE_DIR}/"
cp docker-compose.yml "${STAGE_DIR}/"
cp wordcount.py "${STAGE_DIR}/"
cp kmeans_openfoodfacts.py "${STAGE_DIR}/"
cp build_distribution.sh "${STAGE_DIR}/"
cp input.txt "${STAGE_DIR}/"

mkdir -p "${STAGE_DIR}/data"
cp data/sample_openfoodfacts.csv "${STAGE_DIR}/data/"

if [ -d outputs/openfoodfacts_kmeans ]; then
  mkdir -p "${STAGE_DIR}/outputs"
  cp -R outputs/openfoodfacts_kmeans "${STAGE_DIR}/outputs/"
fi

find "${STAGE_DIR}" -name ".DS_Store" -delete
find "${STAGE_DIR}" -name ".*.crc" -delete

(
  cd "${DIST_DIR}"
  zip -qr "${PROJECT_NAME}.zip" "${PROJECT_NAME}"
)

echo "Created ${ZIP_PATH}"
