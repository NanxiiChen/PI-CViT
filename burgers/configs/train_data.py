from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "cvit"
    train_data_path = "./data/burgers/burgers_training.npz"
    data_dir = "./data/burgers"
    save_dir = "/root/autodl-tmp/tf-logs/burgers/cvit/data_driven"
    ckpt = None
    target_ts = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])  # target time steps for evaluation
    lx = 1.0
    ly = 1.0
    length_scale = 0.1
    amplitude = 0.2
    Nx = 64  # number of spatial points in x
    Ny = 64  # number of spatial points in y
    spatial_domain = ((0, lx), (0, ly))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    active_loss_names = ("data", "pde", "ic",)
    use_multi_gpu = True # some times `nan` occurs when using single gpu, possibly due to `hessian` computation instability

    Lc = 1.0  # xc = x / Lc
    Tc = 1.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        ## model:encoder
        patch_size=(8, 8),
        grid_size=(64, 64),
        in_channels=2,  # u, v
        emb_dim=256,  # emb_dim for encoder
        depth=2,
        num_heads=8,
        ## model:decoder
        fourier_freq=2.0,
        dec_depth=2,
        dec_num_heads=8,
        dec_emb_dim=256,  # two times of ffe hidden dim (sin, cos)
        dec_mlp_act="gelu",
        num_mlp_layers=2,
        out_dim=2,  # u, v
        layer_norm_eps=1e-5,
        use_time_film=False,
        film_depth=1,
        film_act="gelu",
    )

    use_causality = True
    max_grad_norm = 1.0
    optimizer_name = "soap" # adam. soap
    # `adam`` cannot make it, especially for `ic` term.
    alpha_w = 1.0 # moving average weight for loss balancing

    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=5.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )

    # training hyperparameters
    dataset_size = 32 # how many trajectories to use for training
    total_train_step = 15000
    batch_size = 32
    num_samples = 2048
    
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 500
    resample_every = 25
    warmup_steps = 500 if "pde" in active_loss_names else 0

    # material properties
    nu = 0.01