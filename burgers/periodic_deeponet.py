from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from models.deeponet import Encoder, Decoder, DeepONet

class PeriodicDecoder(eqx.Module):
    original_decoder: Decoder
    lx: float = eqx.field(static=True)
    ly: float = eqx.field(static=True)
    
    def __init__(self, original_decoder: Decoder, 
                 lx: float, ly: float,):
        self.original_decoder = original_decoder
        self.lx = lx
        self.ly = ly
        
    def __call__(self, u, x, t):
        # u: (N_patch, enc_emb_dim)
        # x: (N_query, spatial_dim) -> expected to be (N, 2)
        # t: (N_query, 1)
        
        sinx = jnp.sin(2 * jnp.pi * x[:, 0:1] / self.lx)
        cosx = jnp.cos(2 * jnp.pi * x[:, 0:1] / self.lx)
        siny = jnp.sin(2 * jnp.pi * x[:, 1:2] / self.ly)
        cosy = jnp.cos(2 * jnp.pi * x[:, 1:2] / self.ly)
        
        x_periodic = jnp.concatenate([sinx, cosx, siny, cosy], axis=-1)
        
        return self.original_decoder(u, x_periodic, t)
    
    
class PeriodicDeepONet(eqx.Module):
    encoder: Encoder
    decoder: PeriodicDecoder
    
  
    def __init__(
        self, 
        key, 
        lx=1.0, 
        ly=1.0,
        # Encoder (Branch) args
        in_channels: int = 3,
        grid_size: Tuple[int, int] = (224, 224),
        branch_use_cnn: bool = False,
        branch_conv_channels: int = 32,
        branch_conv_kernel: int = 4,
        branch_conv_stride: int = 4,
        branch_mlp_layers: int = 2,
        branch_mlp_hidden: int = 128,
        # Decoder (Trunk) args
        trunk_mlp_layers: int = 3,
        trunk_mlp_hidden: int = 128,
        trunk_fourier_freq: float = 1.0,
        trunk_emb_dim: int = 128,
        trunk_use_time_film: bool = True,
        # Common args
        basis_dim: int = 128,
        out_dim: int = 2,
        coord_dim: int = 3, # x(2) + t(1)
        act: str = "gelu"
    ):
        
        k_enc, k_dec = jax.random.split(key)
        
        self.encoder = Encoder(
            key=k_enc,
            in_channels=in_channels,
            grid_size=grid_size,
            use_cnn=branch_use_cnn,
            conv_out_channels=branch_conv_channels,
            conv_kernel_size=branch_conv_kernel,
            conv_stride=branch_conv_stride,
            mlp_layers=branch_mlp_layers,
            mlp_hidden_dim=branch_mlp_hidden,
            basis_dim=basis_dim,
            out_dim=out_dim,
            act=act
        )
        
        original_decoder = Decoder(
            key=k_dec,
            coord_dim=coord_dim,
            mlp_layers=trunk_mlp_layers,
            mlp_hidden_dim=trunk_mlp_hidden,
            basis_dim=basis_dim,
            out_dim=out_dim,
            fourier_freq=trunk_fourier_freq,
            emb_dim=trunk_emb_dim,
            use_time_film=trunk_use_time_film,
            act=act
        )
        
        self.decoder = PeriodicDecoder(original_decoder, lx, ly)
        
        
    def __call__(self, u, x, t):
        enc_out = self.encoder(u) # (N_patch, enc_emb_dim)
        dec_out = self.decoder(enc_out, x, t) # (N_query, output_dim)
        return dec_out