from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "cvit"
    data_dir = "./data/ldc"
    save_dir = "/root/autodl-tmp/tf-logs/ldc/cvit"
    # ckpt = save_dir + "/baseline-0126-mlplayer3/model_epoch_23000.eqx"
    ckpt = None
    lx = 1.0
    ly = 1.0
    Nx = 256  # number of spatial points in x
    Ny = 256  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    active_loss_names = ("momentum", "continuity", "bc_walls", "bc_lid", "bc_pressure")
    use_multi_gpu = True # some times `nan` occurs when using single gpu, possibly due to `hessian` computation instability

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
    
    
    Lc = 1.0  # xc = x / Lc
    Tc = 1.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        ## model:encoder
        in_channels=1,  # Re number only
        emb_dim=256,  # emb_dim for encoder
        depth=2,
        num_tokens=1,
        ## model:decoder
        fourier_freq=(2.0, 5.0),
        dec_depth=4,
        dec_num_heads=8,
        dec_emb_dim=256,  # two times of ffe hidden dim (sin, cos)
        dec_mlp_act="gelu",
        num_mlp_layers=2,
        out_dim=3,  # u, v, p
        layer_norm_eps=1e-5,
    )

    use_causality = True
    max_grad_norm = 1.0
    optimizer_name = "soap" # adam not work here
    alpha_w = 1.0 # moving average weight for loss balancing


    # training hyperparameters
    num_epochs = 15000
    initial_lr = 5e-4 if ckpt is None else 1e-5
    decay_every = 500
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 1000
    log_every = 50
    test_every = 500
    resample_coord_every = 50
    resample_u_every = 50
    if ckpt is None:
        warmup_epochs = 0
        reach_max_re_epoch = 5000
    else:
        warmup_epochs = 0
        reach_max_re_epoch = 1

    num_u_samples = 32
    num_pde_samples = 2048
    num_bc_samples = 256

