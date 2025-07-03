# https://gist.github.com/mrsteyk/74ad3ec2f6f823111ae4c90e168505ac

import torch
import torch.nn.functional as F
import torch.nn as nn

from models import register

class PositionalEmbedding(nn.Module):
    def __init__(self, pe_dim=320, out_dim=1280, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = pe_dim
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.f_1 = nn.Linear(pe_dim, out_dim)
        self.f_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)



class AudioEmbedding(nn.Module):
    """1D convolution for audio input embedding"""
    def __init__(self, in_channels, out_channels=320, kernel_size=3) -> None:
        super().__init__()
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)

class AudioUnembedding(nn.Module):
    """1D convolution for audio output"""
    def __init__(self, in_channels=320, out_channels=1, kernel_size=3) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class AudioConvResblock(nn.Module):
    """1D Residual block for audio"""
    def __init__(self, in_features, out_features, t_dim, kernel_size=3) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=kernel_size//2)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv1d(out_features, out_features, kernel_size=kernel_size, padding=kernel_size//2)

        skip_conv = in_features != out_features
        self.f_s = (
            nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
            if skip_conv
            else nn.Identity()
        )

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2) + 1  # [batch, channels, 1]
        t_2 = t[1].unsqueeze(dim=2)      # [batch, channels, 1]

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))

class AudioDownsample(nn.Module):
    """1D downsampling for audio"""
    def __init__(self, in_channels, t_dim, downsample_factor=2) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)
        self.downsample_factor = downsample_factor

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        # 1D average pooling
        avg_pool1d = F.avg_pool1d(gn_1, kernel_size=self.downsample_factor)
        f_1 = self.f_1(avg_pool1d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool1d(x_skip, kernel_size=self.downsample_factor)

class AudioUpsample(nn.Module):
    """1D upsampling for audio"""
    def __init__(self, in_channels, t_dim, upsample_factor=2) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)
        self.upsample_factor = upsample_factor

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2) + 1
        t_2 = t_2.unsqueeze(2)

        gn_1 = F.silu(self.gn_1(x))
        # 1D interpolation upsampling
        upsample = F.interpolate(gn_1, scale_factor=self.upsample_factor, mode='nearest')
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.interpolate(x_skip, scale_factor=self.upsample_factor, mode='nearest')


@register('audio_diffusion_unet')
class AudioDiffusionUNet(nn.Module):
    """
    1D UNet for audio diffusion with dynamic latent conditioning
    
    Handles:
    - x: [batch, 1, samples] - audio waveform (dynamic length)
    - z_dec: [batch, 64, n_frames] - latent conditioning (dynamic length)
    """
    
    def __init__(
        self, 
        in_channels=1,           # Audio channels (mono=1, stereo=2)
        z_dec_channels=64,       # Latent conditioning channels
        c0=128, c1=256, c2=512,  # Channel progression (smaller than image version)
        pe_dim=320, 
        t_dim=1280,
        kernel_size=3
    ) -> None:
        super().__init__()
        
        # Store for dynamic conditioning
        self.z_dec_channels = z_dec_channels
        
        # Audio input embedding
        self.embed_audio = AudioEmbedding(
            in_channels=in_channels, 
            out_channels=c0,
            kernel_size=kernel_size
        )
        
        # Time embedding
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)
        
        # Latent conditioning projection
        if z_dec_channels is not None:
            self.z_dec_proj = nn.Conv1d(z_dec_channels, c0, kernel_size=1)
        
        # Downsampling path
        down_0 = nn.ModuleList([
            AudioConvResblock(c0, c0, t_dim, kernel_size),
            AudioConvResblock(c0, c0, t_dim, kernel_size),
            AudioConvResblock(c0, c0, t_dim, kernel_size),
            AudioDownsample(c0, t_dim),
        ])
        down_1 = nn.ModuleList([
            AudioConvResblock(c0, c1, t_dim, kernel_size),
            AudioConvResblock(c1, c1, t_dim, kernel_size),
            AudioConvResblock(c1, c1, t_dim, kernel_size),
            AudioDownsample(c1, t_dim),
        ])
        down_2 = nn.ModuleList([
            AudioConvResblock(c1, c2, t_dim, kernel_size),
            AudioConvResblock(c2, c2, t_dim, kernel_size),
            AudioConvResblock(c2, c2, t_dim, kernel_size),
            AudioDownsample(c2, t_dim),
        ])
        down_3 = nn.ModuleList([
            AudioConvResblock(c2, c2, t_dim, kernel_size),
            AudioConvResblock(c2, c2, t_dim, kernel_size),
            AudioConvResblock(c2, c2, t_dim, kernel_size),
        ])
        self.down = nn.ModuleList([down_0, down_1, down_2, down_3])

        # Middle layers
        self.mid = nn.ModuleList([
            AudioConvResblock(c2, c2, t_dim, kernel_size),
            AudioConvResblock(c2, c2, t_dim, kernel_size),
        ])

        # Upsampling path
        up_3 = nn.ModuleList([
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioUpsample(c2, t_dim),
        ])
        up_2 = nn.ModuleList([
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 * 2, c2, t_dim, kernel_size),
            AudioConvResblock(c2 + c1, c2, t_dim, kernel_size),
            AudioUpsample(c2, t_dim),
        ])
        up_1 = nn.ModuleList([
            AudioConvResblock(c2 + c1, c1, t_dim, kernel_size),
            AudioConvResblock(c1 * 2, c1, t_dim, kernel_size),
            AudioConvResblock(c1 * 2, c1, t_dim, kernel_size),
            AudioConvResblock(c0 + c1, c1, t_dim, kernel_size),
            AudioUpsample(c1, t_dim),
        ])
        up_0 = nn.ModuleList([
            AudioConvResblock(c0 + c1, c0, t_dim, kernel_size),
            AudioConvResblock(c0 * 2, c0, t_dim, kernel_size),
            AudioConvResblock(c0 * 2, c0, t_dim, kernel_size),
            AudioConvResblock(c0 * 2, c0, t_dim, kernel_size),
        ])
        self.up = nn.ModuleList([up_0, up_1, up_2, up_3])

        # Output layer
        self.output = AudioUnembedding(in_channels=c0, out_channels=in_channels)
    
    def get_last_layer_weight(self):
        return self.output.f.weight

    def condition_with_latents(self, x, z_dec):
        """
        Add latent conditioning to audio features
        
        Args:
            x: [batch, c0, audio_samples] - audio features
            z_dec: [batch, 64, n_frames] - latent conditioning
            
        Returns:
            x: [batch, c0, audio_samples] - conditioned features
        """
        if z_dec is None:
            return x
            
        # Project latents to same channel dimension as audio features
        z_proj = self.z_dec_proj(z_dec)  # [batch, c0, n_frames]
        
        # Interpolate latents to match audio length
        if z_proj.shape[-1] != x.shape[-1]:
            z_proj = F.interpolate(
                z_proj, 
                size=x.shape[-1], 
                mode='nearest'  # or 'linear' for smoother interpolation
            )
        
        # Add latent conditioning to audio features
        return x + z_proj

    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, 1, samples] - audio waveform (any length)
            t: [batch] - diffusion timesteps
            z_dec: [batch, 64, n_frames] - latent conditioning (any length)
        """
        # Embed audio input
        x = self.embed_audio(x)  # [batch, c0, samples]
        
        # Add latent conditioning
        if z_dec is not None:
            x = self.condition_with_latents(x, z_dec)

        # Embed timestep
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)  # [batch, t_dim]

        # Downsampling with skip connections
        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        # Middle layers
        for mid in self.mid:
            x = mid(x, t)

        # Upsampling with skip connections
        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, AudioConvResblock):
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, t)

        # Output
        return self.output(x)

