import torch.nn as nn


from models import register
from .model import Encoder, Decoder, WNConv1d


default_configs = {
    'snake': dict(
        encoder_dim=64,
        encoder_rates=[2, 4, 5, 8],
        latent_dim=64,
        d_in=1,
        activation='snake',
    ),
    'snake': dict(
        encoder_dim=64,
        encoder_rates=[2, 4, 5, 8],
        latent_dim=64,
        d_in=1,
        activation='snakebeta',
    ),
}


@register('dac_encoder')
def make_dac_encoder(config_name, **kwargs):
    encoder_kwargs = default_configs[config_name]
    encoder_kwargs.update(kwargs)
    latent_dim = encoder_kwargs['latent_dim']
    return nn.Sequential(
        Encoder(**encoder_kwargs),
        WNConv1d(latent_dim, latent_dim, kernel_size=1),
    )


@register('vqgan_decoder')
def make_vqgan_decoder(config_name, **kwargs):
    decoder_kwargs = default_configs[config_name]
    decoder_kwargs.update(kwargs)
    latent_dim = decoder_kwargs['latent_dim']
    return nn.Sequential(
        WNConv1d(latent_dim, latent_dim, kernel_size=1),
        Decoder(**decoder_kwargs),
    )
