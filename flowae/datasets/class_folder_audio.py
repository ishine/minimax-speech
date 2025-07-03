import os
import random
from PIL import Image, ImageFile

from datasets import register
from torch.utils.data import Dataset
from torchvision import transforms

import os
import random
from pathlib import Path
from typing import Optional, Callable

from models.ldm.dac.audiotools import AudioSignal
from models.ldm.dac.audiotools.core import util
# Audio file extensions (from audiotools)
AUDIO_EXTS = ('.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.mp4', '.MP4', '.m4a', '.M4A')

@register('class_folder_audio')
class AudioFolder(Dataset):
    """
    Audio dataset that loads audio files from a folder structure.
    Similar to ClassFolder but for audio files.
    
    Expected folder structure:
    root_path/
    ├── class1/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    ├── class2/
    │   ├── audio1.wav
    │   └── ...
    └── ...
    
    Or for single class (no subfolders):
    root_path/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
    """

    def __init__(
        self,
        root_path: str,
        sample_rate: int = 24000,
        duration: float = 2.0,
        num_channels: int = 1,
        random_crop: bool = True,
        loudness_cutoff: float = -40,
        audio_only: bool = False,
        drop_label_p: float = 0.0,
        shuffle: bool = True,
        shuffle_state: int = 0,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        trim_silence: bool = False,
    ):
        """
        Args:
            root_path: Path to audio files
            sample_rate: Target sample rate for audio
            duration: Duration in seconds for audio clips
            num_channels: Number of channels (1 for mono, 2 for stereo)
            random_crop: Whether to randomly crop audio (vs deterministic)
            loudness_cutoff: Minimum loudness threshold for audio selection
            audio_only: If True, return only audio signal. If False, return dict with labels
            drop_label_p: Probability of dropping labels (for unconditional training)
            shuffle: Whether to shuffle files
            shuffle_state: Random state for shuffling
            transform: Additional audio transforms
            normalize: Whether to normalize audio amplitude
            trim_silence: Whether to trim silence from audio
        """
        self.root_path = root_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_channels = num_channels
        self.random_crop = random_crop
        self.loudness_cutoff = loudness_cutoff
        self.audio_only = audio_only
        self.drop_label_p = drop_label_p
        self.transform = transform
        self.normalize = normalize
        self.trim_silence = trim_silence
        
        print(f'Audio root_path: {root_path}')
        
        # Find audio files and labels
        self.files = []

        # Fin all audio in recursive in root_path
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(AUDIO_EXTS):
                    self.files.append(os.path.join(root, file))
        
        
        print(f'Found {len(self.files)} audio files')
        
        # Shuffle files if requested
        if shuffle:
            state = util.random_state(shuffle_state)
            combined = self.files
            state.shuffle(combined)
            self.files = combined

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = self.files[idx]
            
            # Load audio using AudioSignal
            if self.random_crop:
                # Use salient excerpt for random cropping with loudness filtering
                signal = AudioSignal.salient_excerpt(
                    str(file_path),
                    duration=self.duration,
                    loudness_cutoff=self.loudness_cutoff,
                )
            else:
                # Load from beginning or deterministic offset
                signal = AudioSignal(
                    str(file_path),
                    duration=self.duration,
                    offset=0.0,
                )
            
            # Convert to mono/stereo as needed
            if self.num_channels == 1:
                signal = signal.to_mono()
            
            # Resample to target sample rate
            signal = signal.resample(self.sample_rate)
            
            # Ensure duration by padding or trimming
            target_samples = int(self.duration * self.sample_rate)
            if signal.length < target_samples:
                signal = signal.zero_pad_to(target_samples)
            elif signal.length > target_samples:
                signal = signal.truncate_samples(target_samples)
            
            # Optional audio processing
            if self.trim_silence:
                signal = signal.trim_silence()
                # Re-pad if trimming made it too short
                if signal.length < target_samples:
                    signal = signal.zero_pad_to(target_samples)
            
            if self.normalize:
                signal = signal.normalize()
            
            # Clamp audio to [-1, 1] range
            signal.audio_data = signal.audio_data.clamp(-1.0, 1.0)
            
            # Apply additional transforms if provided
            if self.transform is not None:
                # Create a random state for transforms
                state = util.random_state(idx)
                transform_args = self.transform.instantiate(state, signal=signal)
                signal = self.transform(signal, **transform_args)
            
            # print('before process: ', signal.audio_data.shape)
            # Store metadata
            signal.metadata.update(
                    {
                    'file_path': str(file_path),
                    'original_sr': signal.sample_rate,
                    'duration': self.duration,
                }
            )
            
            if self.audio_only:
                return signal
            else:
                return {
                    'signal': signal,
                    'file_path': str(file_path),
                    'idx': idx,
                }
                
        except Exception as e:
            print(f'Error loading audio file {self.files[idx]}: {e}')
            # Return next file on error to avoid crashing training
            return self.__getitem__((idx + 1) % len(self))

    def collate(self, batch):
        """Collate function for DataLoader"""
        if self.audio_only:
            # Batch AudioSignals
            return AudioSignal.batch(batch)
        else:
            # Collate dictionary batch
            return util.collate(batch)