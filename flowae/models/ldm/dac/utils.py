import torch.nn as nn


from models import register
from .model import Encoder, Decoder, WNConv1d


default_configs = {
    'snake': dict(
        d_model=64,
        strides=[2, 4, 5, 8],
        d_latent=64,
        d_in=1,
        activation='snake',
    ),
    'snakebeta': dict(
        d_model=64,
        strides=[2, 4, 5, 8],
        d_latent=64,
        d_in=1,
        activation='snakebeta',
    ),
}


@register('dac_encoder')
def make_dac_encoder(config_name, **kwargs):
    encoder_kwargs = default_configs[config_name]
    encoder_kwargs.update(kwargs)
    d_model = encoder_kwargs['d_model']
    return nn.Sequential(
        Encoder(**encoder_kwargs),
        WNConv1d(d_model, d_model, kernel_size=1),
    )


@register('vqgan_decoder')
def make_vqgan_decoder(config_name, **kwargs):
    decoder_kwargs = default_configs[config_name]
    decoder_kwargs.update(kwargs)
    d_model = decoder_kwargs['d_model']
    return nn.Sequential(
        WNConv1d(d_model, d_model, kernel_size=1),
        Decoder(**decoder_kwargs),
    )
