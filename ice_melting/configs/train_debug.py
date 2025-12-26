from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class Config:
    model_name = "cvit"
    data_dir = "./data/ice_melting/ellipse"
    save_dir = "/root/autodl-tmp/tf-logs/ice_melting/debug"
    spatial_domain = ((-0.5, 0.5), (-0.5, 0.5))# x range, y range, normalized
    temporal_domain = (0.0, 3.0) # t range
    a_range = (20, 40)
    b_range = (20, 40)
    theta_range = (0.0, jnp.pi)
    spatial_domain_phys = ((-50.0, 50.0), (-50.0, 50.0)) # physical spatial domain
    Lc = 100.0 # xc = x / Lc
    Tc = 1.0 # tc = t / Tc
    
    
    # model hyperparameters
    model_params = dict(
        ## model:encoder
        patch_size = (8, 8),
        grid_size = (64, 64),
        in_channels = 1, # phi
        emb_dim = 384, # emb_dim for encoder
        depth = 6,
        num_heads = 8,
        
        ## model:decoder
        fourier_freq = 1.0,
        dec_depth = 2,
        dec_num_heads = 8,
        dec_emb_dim = 256, # two times of ffe hidden dim (sin, cos)
        num_mlp_layers = 1,
        out_dim = 1, # phi,
        layer_norm_eps = 1e-5,
    )
    
    # training hyperparameters
    num_epochs = 10000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    batch_size = 64
    save_every = 200
    log_every = 50

    num_u_samples = 16
    num_pde_samples = 1024
    num_ic_samples = 256
    
    lbd = 5.0
    N_val = 63 # element num in fem, points is N_val + 1
    M_val = 0.1

    @property
    def h_val(self):
        return 100 / self.N_val
    
    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
    
    
    
    

    

    
    

    
    