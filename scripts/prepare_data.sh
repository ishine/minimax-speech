#!/bin/bash

# Data preparation pipeline for Learnable-Speech training

echo "=== Learnable-Speech Data Preparation Pipeline ==="

# Configuration
DATASET_ROOT="/path/to/your/dataset"  # Change this to your dataset path
OUTPUT_DIR="/path/to/processed/data"  # Change this to your output path

# Create output directories
mkdir -p $OUTPUT_DIR/{fsq,dac_latents,lists}

echo "Step 1: Extract FSQ tokens using S3Tokenizer..."
cd speech/tools/S3Tokenizer
pip install .

# Extract FSQ tokens (25Hz)
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    `which s3tokenizer` \
    --root_path $DATASET_ROOT \
    --model speech_tokenizer_v2_25hz \
    --device "cuda" \
    --batch_size 64 \
    --file_list ../../../files_test.txt \
    --skip_existing

echo "Step 2: Extract DAC-VAE latents..."
cd ../../../dac-vae

# Download DAC-VAE checkpoint
wget -O checkpoint.pt "https://github.com/primepake/learnable-speech/releases/download/dac-vae/dac_vae_checkpoint.pt"

# Extract DAC latents
python extract_dac_latents.py \
    --checkpoint checkpoint.pt \
    --config configs/config.yml \
    --root_path $DATASET_ROOT \
    --output_dir $OUTPUT_DIR/dac_latents

echo "Step 3: Create data lists..."
cd ../speech
python tools/create_data_list.py \
    --src_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR/lists

echo "Data preparation completed!"
echo "Your dataset should now have:"
echo "  - Original audio files (.wav)"
echo "  - Text transcriptions (.txt)" 
echo "  - FSQ tokens (*_fsq.pt)"
echo "  - DAC latents (*_latent.pt)"
echo "  - Data list files"
