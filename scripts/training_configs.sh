# Learnable-Speech Training Configuration for Different Environments

# ==== LOCAL TRAINING (Single GPU) ====
# For development and testing

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH=/path/to/learnable-speech:$PYTHONPATH

# Single GPU training
python train.py \
  --train_engine torch_ddp \
  --config config.yaml \
  --train_data ./data/train.list \
  --cv_data ./data/val.list \
  --qwen_pretrain_path ./pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --model_dir ./checkpoints/llm/ \
  --num_workers 4 \
  --prefetch 50 \
  --use_amp \
  --pretrained_model ./pretrained_models/CosyVoice2-0.5B/llm.pt

# ==== MULTI-GPU TRAINING (Local) ====
# For faster training on multiple GPUs

export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nnodes=1 --nproc_per_node=$num_gpus --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  train.py \
  --train_engine torch_ddp \
  --config config.yaml \
  --train_data ./data/train.list \
  --cv_data ./data/val.list \
  --qwen_pretrain_path ./pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --model_dir ./checkpoints/llm/ \
  --num_workers 24 \
  --prefetch 100 \
  --use_amp \
  --pretrained_model ./pretrained_models/CosyVoice2-0.5B/llm.pt

# ==== CLOUD TRAINING (Google Colab/Kaggle) ====
# Optimized for limited resources

export CUDA_VISIBLE_DEVICES="0"
pip install -r requirements.txt

python train.py \
  --train_engine torch_ddp \
  --config config.yaml \
  --train_data ./data/small_train.list \
  --cv_data ./data/small_val.list \
  --qwen_pretrain_path ./pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --model_dir /content/checkpoints/llm/ \
  --num_workers 2 \
  --prefetch 25 \
  --use_amp \
  --pretrained_model ./pretrained_models/CosyVoice2-0.5B/llm.pt \
  --comet_disabled  # Disable logging for simplicity

# ==== HUGGING FACE SPACES TRAINING ====
# For training directly on HF infrastructure

# Note: This requires HF Pro subscription for GPU access
# Use smaller batch sizes and enable checkpointing

python train.py \
  --train_engine torch_ddp \
  --config config_hf.yaml \
  --train_data ./data/hf_train.list \
  --cv_data ./data/hf_val.list \
  --qwen_pretrain_path ./pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --model llm \
  --model_dir /tmp/checkpoints/llm/ \
  --num_workers 1 \
  --prefetch 10 \
  --use_amp \
  --pretrained_model ./pretrained_models/CosyVoice2-0.5B/llm.pt \
  --timeout 1800  # 30 minutes timeout for HF
