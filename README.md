# MiniMax-Speech Technical Implementation

An unofficial implementation based on the MiniMax-Speech technical report, with core components adapted from [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice).

![MiniMax-Speech Architecture](assets/image.png)

## Overview

This repository provides an implementation of the MiniMax-Speech model, featuring a two-stage training approach for high-quality 24kHz audio generation.

## Key Features

- [x] **24kHz Audio Support**: High-quality audio generation at 24kHz sampling rate
- [x] **Flow matching AE**: Flow matching training for autoencoders
- [x] **Immiscible assignment**: Support immiscible adding noise while training
- [x] **Contrastive Flow matching**: Support Contrastive training
- [ ] **Checkpoint release**: Release LLM and Contrastive FM checkpoint
## Architecture

### Stage 1: Audio to Discrete Tokens
Converts raw audio into discrete representations using the FSQ (S3Tokenizer) framework.

### Stage 2: Discrete Tokens to Continuous Latent Space
Maps discrete tokens to a continuous latent space using a Variational Autoencoder (VAE).

> **Note**: This implementation uses standard DAC-VAE instead of Flow-VAE.

## Implementation Pipeline

### 1. Model Training

#### BPE tokens to FSQ tokens
- Based on the FSQ
- Using Auto Regressive to predict the FSQ tokens with learnable speaker extractor

#### FSQ tokens to DAC-VAE latent
- Based on Cosyvoice2 flow matching decoder
- Learns continuous latent representations from discrete tokens

### 2. Feature Extraction

Before training the main model:
1. Extract discrete tokens using the trained FSQ [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)
2. Generate continuous latent representations using the trained DAC-VAE - the pretrained I provided here: [DAC-VAE](https://drive.google.com/file/d/1iwZhPlcdDwvPjeON3bFAeYarsV4ZtI2E/view?usp=sharing)

### 3. Two-Stage Training

Train the models sequentially:
- **Stage 1**: BPE tokens → Discrete FSQ 
- **Stage 2**: Discrete FSQ → DAC-VAE Continuous latent space

## Getting Started

### Prerequisites
```bash
# List your dependencies here
pip install -r requirements.txt
```

### Training Pipeline

1. **Extracting FSQ** (if not using pretrained)
   ```bash
   pip install 
   s3tokenizer --wav_scp data.scp \
            --device "cuda" \
            --output_dir "./data" \
            --batch_size 32 \
            --model "speech_tokenizer_v2_25hz"
   ```

2. **Extracting DAC-VAE latent**
   ```bash
   cd dac-vae
   python inference.py --checkpoint checkpoint.pt --config config.yml
   ```

3. **Stage 1: Auto Regressive Transformer**
   ```bash
   #!/bin/bash
   pretrained_model_dir=./pretrained_models/CosyVoice2-0.5B

   export CUDA_VISIBLE_DEVICES="0"
   num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
   job_id=1986
   dist_backend="nccl"
   num_workers=2
   prefetch=100
   train_engine=torch_ddp
   model=llm

   torchrun --nnodes=1 --nproc_per_node=$num_gpus --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
   train.py \
   --train_engine $train_engine \
   --config config.yaml \
   --train_data data/data.list \
   --cv_data data/data.list \
   --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
   --model $model \
   --model_dir /data/checkpoint/$model/ \
   --num_workers ${num_workers} \
   --prefetch ${prefetch} \
   --pin_memory \
   --use_amp \
   --comet_disabled

   ```

4. **Stage 2: FLow matching decoder**
   ```bash
   #!/bin/bash
   pretrained_model_dir=./pretrained_models/CosyVoice2-0.5B
   export CUDA_VISIBLE_DEVICES="0"
   num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
   job_id=1986
   dist_backend="nccl"
   num_workers=2
   prefetch=100
   train_engine=torch_ddp
   model=llm

   torchrun --nnodes=1 --nproc_per_node=$num_gpus --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
   train.py \
   --train_engine $train_engine \
   --config config.yaml \
   --train_data data/data.list \
   --cv_data data/data.list \
   --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
   --model $model \
   --model_dir /data/checkpoint/$model/ \
   --num_workers ${num_workers} \
   --prefetch ${prefetch} \
   --pin_memory \
   --use_amp \
   --comet_disabled

   ```

## Project Structure
```
minimax-speech/
├── assets/
├── dac-vae/
├── flowae/
├── speech/    
│   ├── llm/
│   ├── flow/
└── README.md
```

## Related Projects

This implementation builds upon several key projects:

- **[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)**: Core model architectures and training pipelines
- **[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)**: Audio tokenization framework
- **MiniMax-Speech**: Original technical report and methodology

## Citation

If you use this code in your research, please cite:

```bibtex
@article{minimax-speech,
  title={MiniMax-Speech},
  author={[MiniMax team]},
  year={[2025]}
  url={https://arxiv.org/pdf/2505.07916}
}

@misc{cosyvoice2,
  title={CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens},
  author={[FunAudioLLM Team, SpeechLab@Tongyi, Alibaba Group]},
  year={2024},
  url={https://github.com/FunAudioLLM/CosyVoice}
}
```

## License

This project follows the licensing terms of its dependencies:
- CosyVoice2 components: [Check CosyVoice2 License](https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE)
- FSQ components: [Apache 2.0 License](https://github.com/xingchensong/S3Tokenizer/blob/main/LICENSE)

## Acknowledgments

- **[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)**: This implementation extensively uses code and architectures from CosyVoice2
- **[FSQ](https://github.com/xingchensong/S3Tokenizer)**: For the FSQ implementation
- **MiniMax team**: For the technical report and methodology
- **FunAudioLLM team**: For the excellent CosyVoice2 codebase

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities.

## Contact

[nguyennhutsam.math@gmail.com, https://www.linkedin.com/in/primepake/]