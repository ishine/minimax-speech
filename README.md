---
title: Learnable Speech
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# Learnable-Speech: High-Quality 24kHz Speech Synthesis

An unofficial implementation based on improvements of CosyVoice with learnable encoder and DAC-VAE.

## Demo

This Space provides a demo interface for the Learnable-Speech model. Currently, it shows a placeholder implementation. To use the actual trained model, you would need to:

1. Train the model using the provided training pipeline
2. Upload the trained checkpoints
3. Replace the placeholder inference code with actual model loading and inference

## Features

- [x] **24kHz Audio Support**: High-quality audio generation at 24kHz sampling rate  
- [x] **Flow matching AE**: Flow matching training for autoencoders  
- [x] **Immiscible assignment**: Support immiscible adding noise while training  
- [x] **Contrastive Flow matching**: Support Contrastive training  
- [ ] **Checkpoint release**: Release LLM and Contrastive FM checkpoint  
- [ ] **MeanFlow**: Meanflow for FM model

## Architecture

### Stage 1: Audio to Discrete Tokens

Converts raw audio into discrete representations using the FSQ (S3Tokenizer) framework.

### Stage 2: Discrete Tokens to Continuous Latent Space

Maps discrete tokens to a continuous latent space using a Variational Autoencoder (VAE).

## Links

- [GitHub Repository](https://github.com/primepake/learnable-speech)
- [Technical Paper](https://arxiv.org/pdf/2505.07916)
- [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)

## Usage

1. Enter text in the text box
2. Select a speaker ID (0-10)
3. Click "Generate Speech" to synthesize audio

**Note**: This is currently a placeholder demo. The actual model requires training first.
