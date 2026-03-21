import jax.numpy as jnp

class Config:
    model_name = "cvit"
    train_data_path = "./data/swe/f10/swe_training.npz"
    data_dir = "./data/swe/f10"
    save_dir = "/root/autodl-tmp/tf-logs/swe/cvit"
    ckpt = None
    target_ts = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])  # target time steps for evaluation
    lx = 1.0
    ly = 1.0
    length_scale = 0.1
    amplitude = 0.5
    Nx = 64  # number of spatial points in x
    Ny = 64  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    active_loss_names = ("data", "ic_h", "ic_uv")
    use_multi_gpu = True

    Lc = 1.0  # xc = x / Lc
    Tc = 1.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        ## model:encoder
        patch_size=(8, 8),
        grid_size=(64, 64),
        in_channels=1,  # u
        emb_dim=256,  # emb_dim for encoder
        depth=2,
        num_heads=8,
        ## model:decoder
        fourier_freq=2.0,
        dec_depth=4,
        dec_num_heads=8,
        dec_emb_dim=384,  # two times of ffe hidden dim (sin, cos)
        dec_mlp_act="gelu",
        num_mlp_layers=1,
        out_dim=3,  # h, u, v
        layer_norm_eps=1e-5,
        use_time_film=False,
        film_depth=2,
        film_act="silu",
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
    dataset_size = 1024 # how many trajectories to use for training
    total_train_step = 40000
    batch_size = 32
    num_samples = 2048
    num_ic_samples = 1024
    physics_on_data = True
    
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 200
    resample_every = 10
    warmup_steps = 500
    
    # material properties
    H_val = 1.0  # water depth
    g_val = 1.0
    f_val = 10.0
