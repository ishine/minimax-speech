import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from pathlib import Path
import argparse

# You'll need to have the DiTo codebase available
import models
from omegaconf import OmegaConf

class DiToInference:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize DiTo model from checkpoint"""
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config
        self.config = OmegaConf.create(ckpt['config'])
        
        # Create model
        self.model = models.make(self.config['model'])
        
        # Load state dict
        self.model.load_state_dict(ckpt['model']['sd'])
        
        # Move to device and set to eval
        self.model = self.model.to(device)
        self.model.eval()
        
        # Setup image transforms based on config
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("Model loaded successfully!")
        
    def reconstruct_image(self, image_path, debug=True):
        """Reconstruct a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        if debug:
            debug_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ])
            debug_image = debug_transform(image)
            debug_image.save('debug_1_resized_cropped.png')
            print("Saved debug_1_resized_cropped.png")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Step 1: Encode to latent
            z = self.model.encode(image_tensor)
            
            # Step 2: Decode to features (in DiTo this is identity)
            z_dec = self.model.decode(z)
            print('z_dec.shape:', z_dec.shape)
            
            # Step 3: Prepare coordinate grids
            # Based on the training code, coord and scale are dummy values
            b, c, h, w = image_tensor.shape
            coord = torch.zeros(b, 2, h, w, device=self.device)
            scale = torch.zeros(b, 2, h, w, device=self.device)
            
            # Step 4: Render using diffusion
            reconstructed = self.model.render(z_dec, coord, scale)
        
        # Denormalize from [-1, 1] to [0, 1]
        reconstructed = (reconstructed * 0.5 + 0.5).clamp(0, 1)
        
        return reconstructed
    
    def save_reconstruction(self, image_path, output_path):
        """Reconstruct and save image"""
        reconstructed = self.reconstruct_image(image_path)
        
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        reconstructed_pil = to_pil(reconstructed.squeeze(0).cpu())
        
        # Save
        reconstructed_pil.save(output_path)
        print(f"Saved reconstruction to {output_path}")
        
    def compare_reconstruction(self, image_path, output_path):
        """Save original and reconstruction side by side"""
        # Get reconstruction
        reconstructed = self.reconstruct_image(image_path)
        
        # Load original at same resolution
        original = Image.open(image_path).convert('RGB')
        original = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])(original).unsqueeze(0)
        
        # Concatenate side by side
        comparison = torch.cat([original, reconstructed.cpu()], dim=3)
        
        # Save
        to_pil = transforms.ToPILImage()
        comparison_pil = to_pil(comparison.squeeze(0))
        comparison_pil.save(output_path)
        print(f"Saved comparison to {output_path}")
        
    def batch_reconstruct(self, image_folder, output_folder, max_images=None):
        """Reconstruct all images in a folder"""
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Get all images
        image_paths = list(image_folder.glob('*.png')) + \
                     list(image_folder.glob('*.jpg')) + \
                     list(image_folder.glob('*.jpeg'))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        for img_path in image_paths:
            output_path = output_folder / f"recon_{img_path.name}"
            self.save_reconstruction(str(img_path), str(output_path))
            
        print("Batch reconstruction complete!")

def main():
    parser = argparse.ArgumentParser(description='DiTo Image Reconstruction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DiTo checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or folder')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path')
    parser.add_argument('--compare', action='store_true',
                        help='Save comparison with original')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire folder')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum images to process in batch mode')
    
    args = parser.parse_args()
    
    # Initialize model
    dito = DiToInference(args.checkpoint, device=args.device)
    
    # Process based on mode
    if args.batch:
        dito.batch_reconstruct(args.input, args.output, args.max_images)
    elif args.compare:
        dito.compare_reconstruction(args.input, args.output)
    else:
        dito.save_reconstruction(args.input, args.output)

# Example usage function for direct Python use
def reconstruct_single_image(checkpoint_path, image_path, output_path):
    """Simple function to reconstruct a single image"""
    dito = DiToInference(checkpoint_path)
    dito.save_reconstruction(image_path, output_path)

if __name__ == "__main__":
    main()

# Usage examples:
# 1. Single image reconstruction:
#    python dito_inference.py --checkpoint ckpt-best.pth --input image.jpg --output recon.jpg
#
# 2. Single image with comparison:
#    python dito_inference.py --checkpoint ckpt-best.pth --input image.jpg --output compare.jpg --compare
#
# 3. Batch processing:
#    python dito_inference.py --checkpoint ckpt-best.pth --input input_folder/ --output output_folder/ --batch
#
# 4. Direct Python usage:
#    reconstruct_single_image('ckpt-best.pth', 'input.jpg', 'output.jpg')