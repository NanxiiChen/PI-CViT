from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
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
        mode_t=8,
        mode_x=16,
        mode_y=16,
        depth=8,
        activation="gelu",
        add_coords=True,
        padding=(10, 0, 0) # padding for (t, x, y)
    )
    time_scheme = "rk4" # rk4 or fd
    use_causality = False
    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=5.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )
    
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
    
    num_epochs = 20000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 500
    resample_u_every = 100
    warmup_epochs = 1000
    
    nu = 0.01

    @property
    def h_val(self):
        return 100 / self.N_val

    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
