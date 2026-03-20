from dataclasses import dataclass
import jax.numpy as jnp

class Config:
    model_name = "deeponet"
    data_dir = "./data/swe/f10"
    save_dir = "/root/autodl-tmp/tf-logs/swe/deeponet"
    # ckpt = "/root/autodl-tmp/tf-logs/swe/cvit/STAGE1-baseline-0202-dim384_heads8_depth4/model_epoch_28000.eqx"
    # ckpt = "/root/autodl-tmp/tf-logs/swe/cvit/STAGE2-baseline-0202-dim384_heads8_depth4/model_epoch_11000.eqx"
    ckpt = None
    # ckpt = "/root/autodl-tmp/tf-logs/swe/cvit/SOTA-0201-decdim384-decheads8-decdepth4/model_epoch_35000.eqx"
    target_ts = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])  # target time steps for evaluation
    lx = 1.0
    ly = 1.0
    length_scale = 0.1
    amplitude = 0.5
    Nx = 64  # number of spatial points in x
    Ny = 64  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    active_loss_names = ("momentum", "continuity", "ic_h", "ic_uv")
    use_multi_gpu = True

    Lc = 1.0  # xc = x / Lc
    Tc = 1.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        # Encoder (Branch) args
        in_channels=1,
        grid_size=(64, 64),
        branch_use_cnn=False,
        branch_conv_channels=32,
        branch_conv_kernel=8,
        branch_conv_stride=1,
        branch_mlp_layers=6,
        branch_mlp_hidden=512,

        # Decoder (Trunk) args
        trunk_mlp_layers=6,
        trunk_mlp_hidden=512,
        trunk_fourier_freq=2.0,
        trunk_emb_dim=512,
        trunk_use_time_film=False,

        # Common args
        basis_dim=512,
        out_dim=3,
        coord_dim=5,              # sinx, cosx, siny, cosy, t
        act="tanh",  
    )

    use_causality = True
    max_grad_norm = 1.0
    optimizer_name = "soap"
    alpha_w = 1.0 # moving average weight for loss balancing

    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-5 if ckpt is None else 1e-2,
        max_eps=1.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )

    # training hyperparameters
    num_epochs = 50000
    initial_lr = 5e-4 if ckpt is None else 1e-5
    decay_every = 500
    decay_rate = 0.95
    min_lr = 1e-6
    save_every = 500
    log_every = 50
    test_every = 500
    resample_coord_every = 50
    resample_u_every = 100
    warmup_epochs = 1000 if ckpt is None else 0

    num_u_samples = 32
    num_pde_samples = 2048
    num_ic_samples = 2048
    num_rar_samples = 0
    num_rar_pools = 0 # too slow to compute huge pool prediction, and no apparent benefit

    # material properties
    H_val = 1.0  # water depth
    g_val = 1.0
    f_val = 10.0
