from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class Config:
    model = "cvit"
    data_dir = "./data/ice_melting/ellipse"
    spatial_domain = [[-0.5, 0.5], [-0.5, 0.5]] # x range, y range, normalized
    temporal_domain = [0.0, 3.0] # t range
    Lc = 100.0 # xc = x / Lc
    Tc = 1.0 # tc = t / Tc
    
    
    # model hyperparameters
    ## model:encoder
    patch_size = (16, 16)
    grid_size = (224, 224)
    in_channels = 2 # phi, c
    emb_dim = 768 # emb_dim for encoder
    depth = 6
    num_heads = 8
    
    ## model:decoder
    fourier_freq = 1.0
    dec_depth = 2
    dec_num_heads = 8
    dec_emb_dim = 256 # two times of ffe hidden dim (sin, cos)
    num_mlp_layers = 1
    out_dim = 2 # phi, c
    lay_norm_eps = 1e-5

    
    # training hyperparameters
    num_epochs = 5000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    batch_size = 64
    save_every = 100
    test_every = 100
    
    lbd = 5.0
    N_val = 63 # element num in fem, points is N_val + 1

    @property
    def h_val(self):
        return 100 / self.N_val
    
    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
    
    
    
    

    

    
    

    
    