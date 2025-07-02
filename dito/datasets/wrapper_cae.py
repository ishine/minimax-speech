import random
from PIL import Image

import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms

import datasets
from datasets import register
from utils.geometry import make_coord_scale_grid


from models.ldm.dac.audiotools import AudioSignal
import numpy as np

from models.ldm.dac.audiotools.data.datasets import AudioDataset, AudioLoader
from models.ldm.dac.audiotools import transforms as tfm


class BaseWrapperCAE:

    def __init__(
        self,
        dataset,
        resize_inp,
        return_gt=True,
        gt_glores_lb=None,
        gt_glores_ub=None,
        gt_patch_size=None,
        p_whole=0.0,
        p_max=0.0
    ):
        self.dataset = datasets.make(dataset)
        self.resize_inp = resize_inp
        self.return_gt = return_gt
        self.gt_glores_lb = gt_glores_lb
        self.gt_glores_ub = gt_glores_ub
        self.gt_patch_size = gt_patch_size
        self.p_whole = p_whole
        self.p_max = p_max
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, image):
        assert image.size[0] == image.size[1]
        ret = {}
        
        inp = image.resize((self.resize_inp, self.resize_inp), Image.LANCZOS)
        inp = self.transform(inp)
        ret.update({'inp': inp})
        if not self.return_gt:
            return ret

        if self.gt_glores_lb is None:
            glo = self.transform(image)
        else:
            if random.random() < self.p_whole:
                r = self.gt_patch_size
            elif random.random() < self.p_max:
                r = min(image.size[0], self.gt_glores_ub)
            else:
                r = random.randint(
                    self.gt_glores_lb,
                    max(self.gt_glores_lb, min(image.size[0], self.gt_glores_ub))
                )
            glo = image.resize((r, r), Image.LANCZOS)
            glo = self.transform(glo)

        p = self.gt_patch_size
        ii = random.randint(0, glo.shape[1] - p)
        jj = random.randint(0, glo.shape[2] - p)
        gt_patch = glo[:, ii: ii + p, jj: jj + p]

        x0, y0 = ii / glo.shape[-2], jj / glo.shape[-1]
        x1, y1 = (ii + p) / glo.shape[-2], (jj + p) / glo.shape[-1]
        coord, scale = make_coord_scale_grid((p, p), range=[[x0, x1], [y0, y1]])
        ret['gt'] = torch.cat([
            gt_patch, # 3 p p
            coord.permute(2, 0, 1), # 2 p p
            scale.permute(2, 0, 1), # 2 p p
        ], dim=0)

        return ret


@register('wrapper_cae')
class WrapperCAE(BaseWrapperCAE, Dataset):
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, dict):
            ret = dict()
            ret.update(self.process(data.pop('image')))
            ret.update(data)
            return ret
        else:
            return self.process(data)


@register('wrapper_cae_iterable')
class WrapperCAE(BaseWrapperCAE, IterableDataset):

    def __iter__(self):
        for data in self.dataset:
            if isinstance(data, dict):
                ret = dict()
                ret.update(self.process(data.pop('image')))
                ret.update(data)
                yield ret
            else:
                yield self.process(data)






class BaseWrapperAudioCAE:
    """Base wrapper for audio Convolutional Autoencoder (CAE) training.
    
    Similar to the image wrapper, but for audio data.
    """
    
    def __init__(
        self,
        dataset,
        sample_rate=24000,
        duration=0.38,  # Duration in seconds
        n_samples=None,  # Alternative: specify exact number of samples
        return_gt=True,
        gt_sample_rate=None,  # Ground truth sample rate (if different)
        mono=True,
        normalize=True,
        return_coords=True,  # Whether to return coordinate grids
    ):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = n_samples or int(duration * sample_rate)
        self.return_gt = return_gt
        self.gt_sample_rate = gt_sample_rate or sample_rate
        self.mono = mono
        self.normalize = normalize
        self.return_coords = return_coords
        
    def process(self, audio_data):
        """Process audio data for DiTo training.
        
        Args:
            audio_data: Dictionary with 'signal' key containing AudioSignal
                       or AudioSignal directly
        """
        ret = {}
        
        # Extract AudioSignal
        if isinstance(audio_data, dict):
            signal = audio_data['signal']
        else:
            signal = audio_data
            
        # Convert to mono if needed
        if self.mono and signal.num_channels > 1:
            signal = signal.to_mono()
            
        # Resample to target sample rate
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)
            
        # Extract fixed duration
        if signal.duration < self.duration:
            # Pad if too short
            signal = signal.zero_pad_to(self.n_samples)
        else:
            # Take random excerpt if too long
            max_start = signal.num_samples - self.n_samples
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                signal = signal[..., start_idx:start_idx + self.n_samples]
            else:
                signal = signal[..., :self.n_samples]
                
        # Normalize audio
        audio_tensor = signal.audio_data  # Shape: [channels, samples]
        if self.normalize:
            # Normalize to [-1, 1]
            max_val = audio_tensor.abs().max()
            if max_val > 0:
                audio_tensor = audio_tensor / max_val
                
        # Create input tensor
        ret['inp'] = audio_tensor
        
        if not self.return_gt:
            return ret
            
       
        ret['gt'] = audio_tensor
            
        return ret


@register('wrapper_audio_cae')
class WrapperAudioCAE(BaseWrapperAudioCAE, Dataset):
    """Dataset wrapper for audio CAE training."""
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self.process(data)


@register('wrapper_audio_cae_iterable')
class WrapperAudioCAEIterable(BaseWrapperAudioCAE, IterableDataset):
    """Iterable dataset wrapper for audio CAE training."""
    
    def __iter__(self):
        for data in self.dataset:
            yield self.process(data)


# Example usage with your existing AudioDataset
def create_dito_audio_dataset(config):
    """Create DiTo audio dataset from config."""
    
    # Create base audio dataset using audiotools
    
    # Setup audio loaders
    train_folders = config.get("train_folders", {})
    
    loader = AudioLoader(
        sources=list(train_folders.values()),
        transform=tfm.Compose(
            tfm.VolumeNorm(("uniform", -20, -10)),
            tfm.RescaleAudio(),
        ),
        ext=['.wav', '.flac', '.mp3'],
    )
    
    # Create base dataset
    base_dataset = AudioDataset(
        loaders=loader,
        sample_rate=config['sample_rate'],
        duration=config['duration'],
        n_examples=config['n_examples'],
        num_channels=1 if config.get('mono', True) else 2,
    )
    
    # Wrap with DiTo wrapper
    dito_dataset = WrapperAudioCAE(
        dataset=base_dataset,
        sample_rate=config['sample_rate'],
        duration=config['duration'],
        mono=config.get('mono', True),
        normalize=True,
        return_coords=True,
    )
    
    return dito_dataset


# For your training config, you would use it like:
"""
datasets:
  train:
    name: wrapper_audio_cae
    args:
      dataset:
        name: audio_dataset  # Your base audio dataset
        args:
          sources: ["/path/to/audio/files"]
          sample_rate: 44100
          duration: 2.0
          n_examples: 10000
      sample_rate: 44100
      duration: 2.0
      mono: true
      normalize: true
      return_coords: true
    loader:
      batch_size: 16
      num_workers: 8
      
  val:
    name: wrapper_audio_cae
    args:
      dataset:
        name: audio_dataset
        args:
          sources: ["/path/to/val/audio/files"]
          sample_rate: 44100
          duration: 2.0
          n_examples: 1000
      sample_rate: 44100
      duration: 2.0
      mono: true
      normalize: true
      return_coords: true
    loader:
      batch_size: 16
      num_workers: 8
"""