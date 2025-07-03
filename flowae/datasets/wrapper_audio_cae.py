import random
from PIL import Image

import torch
from torch.utils.data import Dataset, IterableDataset

from datasets import register
import datasets

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
        self.dataset = datasets.make(dataset)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(duration * sample_rate)
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
            
        # Normalize audio
        audio_tensor = signal.audio_data  # Shape: [channels, samples]

        audio_tensor = audio_tensor.squeeze(0)

        # Create input tensor
        ret['inp'] = audio_tensor
        
        if not self.return_gt:
            return ret
            
       
        ret['gt'] = audio_tensor
        # print('audio_tensor shape: ', audio_tensor.shape)
            
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
