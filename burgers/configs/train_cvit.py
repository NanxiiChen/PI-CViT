from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "cvit"
    data_dir = "./data/burgers"
    save_dir = "/root/autodl-tmp/tf-logs/burgers/cvit"
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
    active_loss_names = ("pde", "ic",)

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
        use_time_film=True,
    )

    use_causality = False
    max_grad_norm = 1.0
    optimizer_name = "adam" # adam. soap
    alpha_w = 1.0 # moving average weight for loss balancing

    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=100,
        step_size=5.0,
        min_mean_weight=0.4,
        max_min_weight=0.99,
    )

    # training hyperparameters
    num_epochs = 15000
    initial_lr = 5e-4
    decay_every = 100
    decay_rate = 0.95
    min_lr = 1e-6
    save_every = 500
    log_every = 50
    test_every = 50
    resample_every = 1

    num_u_samples = 1
    num_pde_samples = 2048
    num_rar_samples = 0
    num_rar_pools = 0 # too slow to compute huge pool prediction, and no apparent benefit

    # material properties
    nu = 0.01

    @property
    def h_val(self):
        return 100 / self.N_val

    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
