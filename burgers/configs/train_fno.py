from dataclasses import dataclass
import jax.numpy as jnp


class Config:
    model_name = "fno"
    data_dir = "./data/burgers"
    save_dir = "/root/autodl-tmp/tf-logs/burgers/fno"
    ckpt = None
    target_ts = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])  # target time steps for evaluation
    lx = 1.0
    ly = 1.0
    length_scale = 0.1
    amplitude = 0.2
    Nx = 64  # number of spatial points in x
    Ny = 64  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # time range, normalized
    use_multi_gpu = True
    Lc = 1.0
    Tc = 1.0
    
    model_params = dict(
        in_channels=2,
        out_channels=2,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=32,
        depth=8,
        activation="gelu",
        add_coords=True,
        padding=(10, 0, 0) # padding for (t, x, y)
    )
    time_scheme = "rk4" # rk4 or fd
    use_causality = True
    
    max_grad_norm = 1.0
    optimizer_name = "soap"
    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=5.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )
    
    num_epochs = 15000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 1000
    log_every = 5
    test_every = 25
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16
    
    nu = 0.01