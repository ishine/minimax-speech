#!/usr/bin/env python3
"""Upload trained Learnable-Speech models to Hugging Face Hub"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import torch
import json
from pathlib import Path

def create_model_card(model_name, training_info):
    """Create a model card for the uploaded model"""
    return f"""---
license: apache-2.0
tags:
- text-to-speech
- speech-synthesis
- learnable-speech
- cosyvoice
- pytorch
pipeline_tag: text-to-speech
library_name: pytorch
---

# Learnable-Speech {model_name.upper()}

This is a trained {model_name} model from the Learnable-Speech project, an unofficial implementation based on improvements of CosyVoice with learnable encoder and DAC-VAE.

## Model Description

- **Model Type**: {model_name.upper()} ({"Language Model" if model_name == "llm" else "Flow Matching Decoder"})
- **Architecture**: {"Qwen2-based transformer for BPE→FSQ token mapping" if model_name == "llm" else "Causal conditional flow matching for FSQ→DAC latent mapping"}
- **Sample Rate**: 24kHz
- **Framework**: PyTorch

## Training Details

{training_info}

## Usage

```python
import torch
from learnable_speech import LearnableSpeech

# Load the model
model = LearnableSpeech.from_pretrained("your-username/learnable-speech-{model_name}")

# Generate speech
text = "Hello, this is Learnable-Speech!"
audio = model.synthesize(text)
```

## Citation

If you use this model, please cite:

```bibtex
@article{{learnable-speech,
  title={{Learnable-Speech}},
  author={{Learnable team}},
  year={{2025}},
  url={{https://arxiv.org/pdf/2505.07916}}
}}
```

## Links

- [GitHub Repository](https://github.com/primepake/learnable-speech)
- [Original Paper](https://arxiv.org/pdf/2505.07916)
- [Hugging Face Space Demo](https://huggingface.co/spaces/mnhatdaous/learnable-speech)
"""

def upload_model_to_hf(checkpoint_path, model_name, repo_name, token=None, private=False):
    """Upload trained model to Hugging Face Hub"""
    
    api = HfApi(token=token)
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository {repo_name} created/found")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False
    
    # Load checkpoint to get training info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        training_info = f"""
- **Training Steps**: {checkpoint.get('step', 'Unknown')}
- **Training Epochs**: {checkpoint.get('epoch', 'Unknown')}
- **Training Framework**: PyTorch DDP with AMP
- **Optimizer**: AdamW
- **Learning Rate**: {checkpoint.get('lr', 'Unknown')}
"""
    except Exception as e:
        print(f"⚠️  Could not load checkpoint info: {e}")
        training_info = "Training information not available"
    
    # Create model card
    model_card = create_model_card(model_name, training_info)
    
    # Save model card to temporary file
    with open(f"README_{model_name}.md", "w") as f:
        f.write(model_card)
    
    try:
        # Upload checkpoint
        upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_name,
            token=token
        )
        print(f"✅ Model checkpoint uploaded")
        
        # Upload model card
        upload_file(
            path_or_fileobj=f"README_{model_name}.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token
        )
        print(f"✅ Model card uploaded")
        
        # Create and upload config
        config = {
            "model_type": "learnable_speech",
            "architecture": model_name,
            "sample_rate": 24000,
            "framework": "pytorch"
        }
        
        with open(f"config_{model_name}.json", "w") as f:
            json.dump(config, f, indent=2)
        
        upload_file(
            path_or_fileobj=f"config_{model_name}.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            token=token
        )
        print(f"✅ Config uploaded")
        
        # Cleanup
        os.remove(f"README_{model_name}.md")
        os.remove(f"config_{model_name}.json")
        
        print(f"🎉 Model successfully uploaded to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload Learnable-Speech models to Hugging Face")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing trained checkpoints")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repositories private")
    parser.add_argument("--models", nargs="+", choices=["llm", "flow", "both"], default=["both"],
                       help="Which models to upload")
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ Please provide Hugging Face token via --token or HF_TOKEN env var")
        return
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    models_to_upload = []
    if "both" in args.models:
        models_to_upload = ["llm", "flow"]
    else:
        models_to_upload = args.models
    
    success_count = 0
    
    for model_name in models_to_upload:
        print(f"\n🚀 Uploading {model_name.upper()} model...")
        
        # Find latest checkpoint
        model_dir = checkpoint_dir / model_name
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            continue
            
        checkpoint_files = list(model_dir.glob("*.pt"))
        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {model_dir}")
            continue
            
        # Get the latest checkpoint (by modification time)
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"📁 Using checkpoint: {latest_checkpoint}")
        
        # Upload to HF
        repo_name = f"{args.username}/learnable-speech-{model_name}"
        success = upload_model_to_hf(
            checkpoint_path=str(latest_checkpoint),
            model_name=model_name,
            repo_name=repo_name,
            token=token,
            private=args.private
        )
        
        if success:
            success_count += 1
    
    print(f"\n🎉 Upload complete! {success_count}/{len(models_to_upload)} models uploaded successfully")
    
    if success_count > 0:
        print("\n📝 Next steps:")
        print("1. Update your Gradio app to use the uploaded models")
        print("2. Test the models in your Hugging Face Space")
        print("3. Share your trained models with the community!")

if __name__ == "__main__":
    main()
