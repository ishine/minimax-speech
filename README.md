# MiniMax-Speech Technical Implementation

An unofficial implementation based on the MiniMax-Speech technical report, with core components adapted from [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice).

![MiniMax-Speech Architecture](assets/image.png)

## Overview

This repository provides an implementation of the MiniMax-Speech model, featuring a two-stage training approach for high-quality 24kHz audio generation.

## Key Features

- **24kHz Audio Support**: High-quality audio generation at 24kHz sampling rate
- **Two-Stage Architecture**: Optimized training pipeline with discrete and continuous representations
- **Modular Design**: Separate components for audio codec and variational autoencoder
- **CosyVoice2 Integration**: Leverages proven components from the CosyVoice2 framework

## Architecture

### Stage 1: Audio to Discrete Tokens
Converts raw audio into discrete representations using the DAC (Descript Audio Codec) framework.

### Stage 2: Discrete Tokens to Continuous Latent Space
Maps discrete tokens to a continuous latent space using a Variational Autoencoder (VAE).

> **Note**: This implementation uses standard DAC-VAE instead of Flow-VAE.

## Implementation Pipeline

### 1. Model Training

#### DAC Codec
- Based on the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
- Provides efficient audio tokenization
- Utilizes CosyVoice2's optimized training pipeline

#### DAC-VAE
- Train using `train_dac_vae.py`
- Learns continuous latent representations from discrete tokens
- Architecture adapted from CosyVoice2's VAE implementation

### 2. Feature Extraction

Before training the main model:
1. Extract discrete tokens using the trained DAC codec
2. Generate continuous latent representations using the trained DAC-VAE

### 3. Two-Stage Training

Train the models sequentially:
- **Stage 1**: Audio → Discrete token modeling
- **Stage 2**: Discrete token → Continuous latent space modeling

## Getting Started

### Prerequisites
```bash
# List your dependencies here
pip install -r requirements.txt
```

### Training Pipeline

1. **Train DAC Codec** (if not using pretrained)
   ```bash
   # Add training command
   ```

2. **Train DAC-VAE**
   ```bash
   python train_dac_vae.py --config configs/dac_vae.yaml
   ```

3. **Extract Features**
   ```bash
   # Add feature extraction commands
   ```

4. **Train MiniMax-Speech**
   ```bash
   # Add main training command
   ```

## Project Structure
```
minimax-speech/
├── assets/
│   └── image.png
├── configs/
│   └── dac_vae.yaml
├── models/
│   ├── dac_codec/
│   └── dac_vae/
├── cosyvoice/          # Components from CosyVoice2
│   ├── flow/
│   ├── transformer/
│   └── utils/
├── train_dac_vae.py
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
- DAC components: [Apache 2.0 License](https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE)
- Original contributions: [Specify your license here]

## Acknowledgments

- **[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)**: This implementation extensively uses code and architectures from CosyVoice2
- **[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)**: For the DAC implementation
- **MiniMax team**: For the technical report and methodology
- **FunAudioLLM team**: For the excellent CosyVoice2 codebase

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Your contact information or links to issues/discussions]