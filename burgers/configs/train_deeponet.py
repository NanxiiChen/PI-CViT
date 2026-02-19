from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "deeponet"
    data_dir = "./data/burgers"
    save_dir = "/root/autodl-tmp/tf-logs/burgers/deeponet"
    # ckpt = save_dir + "/20260115-204856/model_epoch_3000.eqx"
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
    use_multi_gpu = True # some times `nan` occurs when using single gpu, possibly due to `hessian` computation instability

    Lc = 1.0  # xc = x / Lc
    Tc = 1.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        # Encoder (Branch) args
        in_channels=2,
        grid_size=(64, 64),
        branch_use_cnn=False,
        branch_conv_channels=32,
        branch_conv_kernel=8,
        branch_conv_stride=1,
        branch_mlp_layers=8,
        branch_mlp_hidden=256,

        # Decoder (Trunk) args
        trunk_mlp_layers=8,
        trunk_mlp_hidden=256,
        trunk_fourier_freq=2.0,
        trunk_emb_dim=256,
        trunk_use_time_film=True,

        # Common args
        basis_dim=256,
        out_dim=2,
        coord_dim=5,              # sinx, cosx, siny, cosy, t
        act="tanh",
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
    num_epochs = 20000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 500
    resample_coord_every = 100
    resample_u_every = 100
    warmup_epochs = 1500

    num_u_samples = 32
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
