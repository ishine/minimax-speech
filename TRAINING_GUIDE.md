# 🎤 Learnable-Speech Training Quick Start Guide

This guide will help you train the Learnable-Speech model from scratch and deploy it on Hugging Face.

## 📋 Prerequisites

1. **Hardware Requirements**:
   - GPU with at least 8GB VRAM (16GB+ recommended)
   - 32GB+ RAM
   - 100GB+ storage space

2. **Software Requirements**:
   - Python 3.10+
   - CUDA 11.8+
   - PyTorch 2.0+

## 🚀 Step-by-Step Training Process

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/primepake/learnable-speech.git
cd learnable-speech

# Install dependencies
pip install -r requirements.txt

# Install S3Tokenizer
cd speech/tools/S3Tokenizer
pip install .
cd ../../..
```

### Step 2: Download Prerequisites

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Download pretrained models
./scripts/download_pretrained.sh
```

### Step 3: Prepare Your Dataset

```bash
# Organize your dataset like this:
# dataset_root/
# ├── speaker1_001.wav
# ├── speaker1_001.txt
# ├── speaker1_002.wav
# ├── speaker1_002.txt
# └── ...

# Update DATASET_ROOT in the script
export DATASET_ROOT="/path/to/your/dataset"
export OUTPUT_DIR="/path/to/processed/data"

# Run data preparation
./scripts/prepare_data.sh
```

### Step 4: Train the Models

```bash
# Option A: Train full pipeline (recommended)
./scripts/train_full_pipeline.sh

# Option B: Train stages separately
./speech/llm_run.sh    # Stage 1: LLM
./speech/flow_run.sh   # Stage 2: Flow
```

### Step 5: Upload to Hugging Face

```bash
# Get your HF token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

# Upload trained models
python scripts/upload_to_hf.py \
  --checkpoint_dir ./checkpoints \
  --username your_hf_username \
  --models both
```

### Step 6: Update Gradio App

```python
# Update app.py to use your trained models
from huggingface_hub import hf_hub_download
import torch

# Download your trained models
llm_path = hf_hub_download(
    repo_id="your_username/learnable-speech-llm",
    filename="pytorch_model.bin"
)
flow_path = hf_hub_download(
    repo_id="your_username/learnable-speech-flow", 
    filename="pytorch_model.bin"
)

# Load and use models in your synthesis function
def synthesize_speech(text, speaker_id=0):
    # Replace placeholder with actual model inference
    # ... your inference code here ...
    pass
```

## 🎯 Training Configurations

### For Different Environments

1. **Local Development** (Single GPU):

   ```bash
   export CUDA_VISIBLE_DEVICES="0"
   python speech/train.py --config speech/config.yaml --model llm ...
   ```

2. **Multi-GPU Training**:

   ```bash
   export CUDA_VISIBLE_DEVICES="0,1,2,3"
   torchrun --nproc_per_node=4 speech/train.py ...
   ```

3. **Cloud Training** (Google Colab/Kaggle):

   ```python
   # Use config_hf.yaml for resource-constrained environments
   !python speech/train.py --config speech/config_hf.yaml ...
   ```

4. **Hugging Face Spaces**:

   ```bash
   # For direct training on HF infrastructure
   python speech/train.py --config speech/config_hf.yaml --timeout 1800 ...
   ```

## 📊 Monitoring Training

1. **Comet ML** (Recommended):

   ```bash
   # Set up Comet ML for experiment tracking
   export COMET_API_KEY="your_api_key"
   # Training will automatically log to Comet
   ```

2. **Tensorboard**:

   ```bash
   tensorboard --logdir ./tensorboard
   ```

3. **Command Line**:

   ```bash
   # Monitor log files
   tail -f checkpoints/llm/train.log
   ```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training (`--use_amp`)

2. **Slow Training**:
   - Increase num_workers for data loading
   - Use multiple GPUs with DDP
   - Optimize data preprocessing

3. **Model Not Converging**:
   - Check learning rate
   - Verify data preprocessing
   - Use pretrained checkpoints

### Performance Tips

1. **Data Loading Optimization**:

   ```yaml
   # In config.yaml
   num_workers: 24
   prefetch: 100
   pin_memory: true
   ```

2. **Memory Optimization**:

   ```bash
   # Use gradient checkpointing
   --use_amp --accum_grad 2
   ```

3. **Speed Optimization**:

   ```bash
   # Compile model for faster training (PyTorch 2.0+)
   export TORCH_COMPILE=1
   ```

## 📈 Expected Training Times

| Configuration | LLM Training | Flow Training | Total |
|---------------|--------------|---------------|-------|
| Single RTX 4090 | 2-3 days | 1-2 days | 3-5 days |
| 4x RTX 4090 | 12-18 hours | 6-12 hours | 1-2 days |
| 8x A100 | 6-8 hours | 3-4 hours | 9-12 hours |

## 🎉 Success Criteria

Your training is successful when:

1. **LLM Stage**: Perplexity < 2.0, Token accuracy > 95%
2. **Flow Stage**: Reconstruction loss < 0.1, Mel spectral loss < 0.05
3. **Audio Quality**: Generated samples sound natural and intelligible

## 📚 Additional Resources

- [Training Logs Analysis](docs/training_analysis.md)
- [Hyperparameter Tuning Guide](docs/hyperparameters.md)
- [Deployment Best Practices](docs/deployment.md)
- [Community Discord](https://discord.gg/learnable-speech)
