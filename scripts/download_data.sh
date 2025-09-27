#!/bin/bash
# scripts/download_data.sh
# Script to download and extract the raw dataset from Google Drive

set -e

# Create target directory
mkdir -p data/raw

echo ">>> Downloading raw dataset from Google Drive..."
# Requires: pip install gdown
gdown --id 13ALTcd_Kx6tV4cET-LiYq4jwPIXASDwJ -O data/raw/raw_data.zip

echo ">>> Extracting raw dataset..."
unzip -o data/raw/raw_data.zip -d data/raw

echo ">>> Done! Raw data is available in data/raw/"
