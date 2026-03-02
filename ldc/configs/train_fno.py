from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "fno2d"
    data_dir = "./data/ldc"
    save_dir = "/root/autodl-tmp/tf-logs/ldc/fno"
    ckpt = None
    lx = 1.0
    ly = 1.0
    Nx = 256  # number of spatial points in x
    Ny = 256  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # time range, normalized
    active_loss_names = ("momentum", "continuity", "bc_walls", "bc_lid", "bc_pressure")
    use_multi_gpu = True
    re_range = (50, 1000)  # reynolds number range for data generation
    # ! sample in log space for better generalization
    # ! use log(Re) in neural network input
    re_range_log = (jnp.log(re_range[0]), jnp.log(re_range[1])) 
    re_range_initial = (50, 100) # start from lower Re for better convergence
    evaluate_on_re = [50, 100, 200, 500, 1000]  # Re numbers for evaluation

    @classmethod
    def normalize_re(cls, re):
        """normalize re to [0, 1]"""
        re_log = jnp.log(re)
        re_min, re_max = cls.re_range_log
        return (re_log - re_min) / (re_max - re_min)
    
    @classmethod
    def denormalize_re(cls, re_normalized):
        """denormalize re from [0, 1] to original range"""
        re_min, re_max = cls.re_range_log
        re_log = re_normalized * (re_max - re_min) + re_min
        return jnp.exp(re_log)
    
    Lc = 1.0
    Tc = 1.0
    
    model_params = dict(
        in_channels=1,
        out_channels=3,
        modes_x=8,
        modes_y=8,
        width=64,
        depth=4,
        activation="gelu",
        add_coords=True,
        padding=(10, 10) # padding for (t, x, y)
    )
    


    max_grad_norm = 1.0
    optimizer_name = "soap"

    num_epochs = 20000
    initial_lr = 5e-4 if ckpt is None else 1e-5
    decay_every = 500
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 50
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16
    reach_max_re_epoch=2000
    
    H_val = 1.0  # water depth
    g_val = 1.0
    f_val = 10.0
