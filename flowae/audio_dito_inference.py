import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import soundfile as sf
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Import models
import models
from models.ldm.dac.audiotools import AudioSignal


class AudioDiToInference:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize Audio DiTo model from checkpoint"""
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
        
        # Get audio parameters from config
        self.sample_rate = self.config.get('sample_rate', 24000)
        self.mono = self.config.get('mono', True)
        
        print(f"Model loaded successfully!")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Mono: {self.mono}")
        
    def load_audio(self, audio_path, duration=None, offset=0.0):
        """Load audio file using AudioSignal
        
        Args:
            audio_path: Path to audio file
            duration: Duration in seconds (None for full audio)
            offset: Start offset in seconds
        """
        # Load audio using AudioSignal
        if duration is not None:
            signal = AudioSignal(
                str(audio_path),
                duration=duration,
                offset=offset,
            )
        else:
            # Load full audio
            signal = AudioSignal(str(audio_path))
        
        # Convert to mono if needed
        if self.mono and signal.num_channels > 1:
            signal = signal.to_mono()
        
        # Resample to model sample rate
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)
        
        # Normalize
        signal = signal.normalize()
        
        # Clamp to [-1, 1]
        signal.audio_data = signal.audio_data.clamp(-1.0, 1.0)
        
        return signal
    
    def save_audio(self, reconstructed, output_path):
        """Save AudioSignal to file"""
        # Get audio data
        print('shape of reconstructed: ', reconstructed.shape)
        sf.write(output_path, reconstructed, self.sample_rate)
        print(f"Saved audio to {output_path}")
    
    def reconstruct_audio(self, audio_path, num_steps=50, save_latent=False):
        """Reconstruct entire audio file at once
        
        Args:
            audio_path: Path to audio file
            num_steps: Number of diffusion steps
            save_latent: Whether to return the latent representation
        """
        # Load full audio without duration limit
        signal = self.load_audio(audio_path, duration=None, offset=0.0)
        
        # Get audio tensor
        audio_tensor = signal.audio_data  # [channels, samples]
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)  # [samples] for mono
        
        # Add batch dimension
        audio_tensor = audio_tensor.to(self.device)  # [1, samples]
        
        print(f"Input shape: {audio_tensor.shape}")
        print(f"Full audio duration: {audio_tensor.shape[-1] / self.sample_rate:.2f}s")
        
        with torch.no_grad():
            # Prepare data dict
            data = {'inp': audio_tensor}
            
            # Step 1: Encode to latent
            print('shape of audio_tensor: ', audio_tensor.shape)
            z = self.model.encode(audio_tensor)
            print(f"Latent shape: {z.shape}")
            
            # Step 2: Decode latent (if model has separate decode step)
            if hasattr(self.model, 'decode'):
                z_dec = self.model.decode(z)
            else:
                z_dec = z
            print(f"Decoded latent shape: {z_dec.shape}")
            
            # Step 3: Prepare dummy coordinates (based on training code)
            b, *_ = audio_tensor.shape

            
            # Step 4: Render using diffusion
            if hasattr(self.model, 'render'):
                # Render expects z_dec, coord, scale
                print('using render diffusion model')
                reconstructed = self.model.render(z_dec)
            else:
                # Alternative: direct decode if render not available
                reconstructed = self.model(data, mode='pred')
        
        # Remove batch dimension
        reconstructed = reconstructed.squeeze(0).squeeze(0).cpu().numpy()  # [samples]

        print('shape of reconstructed: ', reconstructed.shape)
    
        
        if save_latent:
            return reconstructed, z.cpu()
        else:
            return reconstructed
    
    def save_reconstruction(self, audio_path, output_path, num_steps=50):
        """Reconstruct and save entire audio file"""
        reconstructed = self.reconstruct_audio(audio_path, num_steps)
        self.save_audio(reconstructed, output_path)
    
    def compare_reconstruction(self, audio_path, output_path, num_steps=50):
        """Save original and reconstruction concatenated"""
        # Load original full audio
        original = self.load_audio(audio_path, duration=None, offset=0.0)
        
        # Get reconstruction of full audio
        reconstructed = self.reconstruct_audio(audio_path, num_steps)
        
        # Add 0.5 second silence between clips
        silence_samples = int(0.5 * self.sample_rate)
        silence_data = torch.zeros(1, silence_samples)
        
        # Concatenate: original -> silence -> reconstruction
        concat_data = torch.cat([
            original.audio_data.cpu(),
            silence_data,
            reconstructed.audio_data.cpu()
        ], dim=1)
        
        # Create concatenated signal
        comparison = AudioSignal(
            concat_data,
            sample_rate=self.sample_rate
        )
        
        self.save_audio(comparison, output_path)
        print(f"Saved comparison (original + reconstruction) to {output_path}")
    
    def visualize_latent(self, audio_path, output_path):
        """Visualize the latent representation of full audio"""
        # Get latent
        _, z = self.reconstruct_audio(audio_path, save_latent=True)
        
        z_np = z.squeeze(0).numpy()  # Remove batch dimension
        
        # Create visualization
        if z_np.ndim == 2:  # [channels, frames]
            n_channels = z_np.shape[0]
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
            
            if n_channels == 1:
                axes = [axes]
            
            for i in range(n_channels):
                im = axes[i].imshow(
                    z_np[i:i+1], 
                    aspect='auto', 
                    cmap='coolwarm',
                    interpolation='nearest'
                )
                axes[i].set_title(f'Latent Channel {i+1}')
                axes[i].set_xlabel('Time Frames')
                axes[i].set_ylabel('Feature')
                plt.colorbar(im, ax=axes[i])
        else:  # 1D latent
            plt.figure(figsize=(12, 4))
            plt.plot(z_np.T)
            plt.title('Latent Representation')
            plt.xlabel('Time Frames')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved latent visualization to {output_path}")
    
    def batch_reconstruct(self, audio_folder, output_folder, max_files=None, num_steps=50):
        """Reconstruct all audio files in a folder (full audio)"""
        audio_folder = Path(audio_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_paths = []
        for ext in audio_extensions:
            audio_paths.extend(audio_folder.glob(f'*{ext}'))
            audio_paths.extend(audio_folder.glob(f'*{ext.upper()}'))
        
        if max_files:
            audio_paths = audio_paths[:max_files]
        
        print(f"Processing {len(audio_paths)} audio files...")
        
        for audio_path in audio_paths:
            output_path = output_folder / f"recon_{audio_path.stem}.wav"
            try:
                self.save_reconstruction(
                    str(audio_path), str(output_path), 
                    num_steps=num_steps
                )
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        print("Batch reconstruction complete!")


def main():
    parser = argparse.ArgumentParser(description='Audio DiTo Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Audio DiTo checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio path or folder')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path')
    parser.add_argument('--compare', action='store_true',
                        help='Save comparison with original')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire folder')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize latent representation')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to process in batch mode')
    
    args = parser.parse_args()
    
    # Initialize model
    audio_dito = AudioDiToInference(args.checkpoint, device=args.device)
    
    # Process based on mode
    if args.batch:
        # Batch processing
        audio_dito.batch_reconstruct(
            args.input, args.output, 
            max_files=args.max_files,
            num_steps=args.steps
        )
    elif args.visualize:
        # Visualize latent
        audio_dito.visualize_latent(
            args.input, args.output
        )
    elif args.compare:
        # Save comparison
        audio_dito.compare_reconstruction(
            args.input, args.output,
            num_steps=args.steps
        )
    else:
        # Single reconstruction
        audio_dito.save_reconstruction(
            args.input, args.output,
            num_steps=args.steps
        )


# Example usage function for direct Python use
def reconstruct_single_audio(checkpoint_path, audio_path, output_path):
    """Simple function to reconstruct a single audio file"""
    audio_dito = AudioDiToInference(checkpoint_path)
    audio_dito.save_reconstruction(audio_path, output_path)


if __name__ == "__main__":
    main()


# Usage examples:
# 1. Single audio reconstruction (full audio):
#    python audio_dito_inference.py --checkpoint ckpt-best.pth --input audio.wav --output recon.wav
#
# 2. Save comparison (original + reconstruction):
#    python audio_dito_inference.py --checkpoint ckpt-best.pth --input audio.wav --output compare.wav --compare
#
# 3. Batch processing (reconstruct all audio files in folder):
#    python audio_dito_inference.py --checkpoint ckpt-best.pth --input audio_folder/ --output output_folder/ --batch
#
# 4. Visualize latent representation:
#    python audio_dito_inference.py --checkpoint ckpt-best.pth --input audio.wav --output latent.png --visualize
#
# 5. Use fewer diffusion steps for faster inference:
#    python audio_dito_inference.py --checkpoint ckpt-best.pth --input audio.wav --output recon.wav --steps 25