#!/bin/bash

# Complete Learnable-Speech Training Pipeline
# This script trains both LLM and Flow models sequentially

set -e  # Exit on any error

echo "🎤 Starting Learnable-Speech Training Pipeline"
echo "=============================================="

# Configuration
DATASET_ROOT="${DATASET_ROOT:-/data/dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
PRETRAINED_DIR="${PRETRAINED_DIR:-./pretrained_models/CosyVoice2-0.5B}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Create checkpoint directories
mkdir -p $CHECKPOINT_DIR/{llm,flow}

# Check prerequisites
echo "📋 Checking prerequisites..."
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "❌ Pretrained models not found. Please run scripts/download_pretrained.sh first"
    exit 1
fi

if [ ! -f "./data/train.list" ]; then
    echo "❌ Training data not found. Please run scripts/prepare_data.sh first"
    exit 1
fi

# Set environment
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Adjust as needed
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "🚀 Starting Stage 1: LLM Training (BPE → FSQ tokens)"
echo "=================================================="

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  speech/train.py \
  --train_engine torch_ddp \
  --config speech/config.yaml \
  --train_data ./data/train.list \
  --cv_data ./data/val.list \
  --qwen_pretrain_path $PRETRAINED_DIR/CosyVoice-BlankEN \
  --model llm \
  --model_dir $CHECKPOINT_DIR/llm/ \
  --num_workers 24 \
  --prefetch 100 \
  --use_amp \
  --pretrained_model $PRETRAINED_DIR/llm.pt \
  --comet_project "learnable-speech" \
  --comet_experiment_name "llm-training-$(date +%Y%m%d-%H%M%S)"

if [ $? -eq 0 ]; then
    echo "✅ Stage 1 (LLM) training completed successfully!"
else
    echo "❌ Stage 1 (LLM) training failed!"
    exit 1
fi

echo "🚀 Starting Stage 2: Flow Training (FSQ → DAC latents)"
echo "====================================================="

# Find the latest LLM checkpoint
LATEST_LLM_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/llm/*.pt | head -1)
echo "Using LLM checkpoint: $LATEST_LLM_CHECKPOINT"

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --rdzv_id=1987 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
  speech/train.py \
  --train_engine torch_ddp \
  --config speech/config.yaml \
  --train_data ./data/train.list \
  --cv_data ./data/val.list \
  --qwen_pretrain_path $PRETRAINED_DIR/CosyVoice-BlankEN \
  --model flow \
  --model_dir $CHECKPOINT_DIR/flow/ \
  --num_workers 24 \
  --prefetch 100 \
  --use_amp \
  --pretrained_model $PRETRAINED_DIR/flow.pt \
  --comet_project "learnable-speech" \
  --comet_experiment_name "flow-training-$(date +%Y%m%d-%H%M%S)"

if [ $? -eq 0 ]; then
    echo "✅ Stage 2 (Flow) training completed successfully!"
else
    echo "❌ Stage 2 (Flow) training failed!"
    exit 1
fi

echo "🎉 Training pipeline completed successfully!"
echo "=========================================="
echo "Trained models saved in: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "1. Test your models with inference scripts"
echo "2. Upload checkpoints to Hugging Face Hub"
echo "3. Update the Gradio app with trained models"

# Create a summary file
cat > $CHECKPOINT_DIR/training_summary.txt << EOF
Learnable-Speech Training Summary
Generated: $(date)

Dataset: $DATASET_ROOT
LLM Checkpoint: $(ls -t $CHECKPOINT_DIR/llm/*.pt | head -1)
Flow Checkpoint: $(ls -t $CHECKPOINT_DIR/flow/*.pt | head -1)

Configuration:
- GPUs: $NUM_GPUS
- Batch Size: $BATCH_SIZE
- Mixed Precision: Enabled
- Framework: PyTorch DDP

Training completed successfully!
EOF

echo "📄 Training summary saved to: $CHECKPOINT_DIR/training_summary.txt"
