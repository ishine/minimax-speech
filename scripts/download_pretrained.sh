#!/bin/bash

# Create pretrained models directory
mkdir -p pretrained_models/CosyVoice2-0.5B

echo "Downloading CosyVoice2 pretrained models..."

# Download CosyVoice2 models (you'll need to get these from the official release)
# Replace these URLs with actual download links when available
wget -O pretrained_models/CosyVoice2-0.5B/llm.pt "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/llm.pt"
wget -O pretrained_models/CosyVoice2-0.5B/flow.pt "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/flow.pt"

# Download Qwen pretrained model
mkdir -p pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN
echo "Download Qwen model manually from: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B"

echo "Pretrained models downloaded!"
