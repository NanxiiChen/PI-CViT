from dataclasses import dataclass
import jax
import jax.numpy as jnp


class Config:
    model_name = "deeponet"
    data_dir = "./data/ice_melting/ellipse_refined_dt"
    save_dir = "/root/autodl-tmp/tf-logs/ice_melting/ellipse/deeponet"
    ckpt = None
    target_ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    spatial_domain = ((-0.5, 0.5), (-0.5, 0.5))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    a_range = (20, 40)
    b_range = (20, 40)
    theta_range = (0.0, jnp.pi)
    spatial_domain_phys = ((-50.0, 50.0), (-50.0, 50.0))  # physical spatial domain
    Lc = 100.0  # xc = x / Lc
    Tc = 3.0  # tc = t / Tc
    active_loss_names = ("pde", "ic", "irr",)

    # model hyperparameters
    model_params = dict(
        # Encoder (Branch) args
        in_channels=1,
        grid_size=(64, 64),
        branch_use_cnn = False,
        branch_conv_channels=64,
        branch_conv_kernel=3,
        branch_conv_stride=1,
        branch_mlp_layers=6,
        branch_mlp_hidden=256,

        # Decoder (Trunk) args
        trunk_mlp_layers=6,
        trunk_mlp_hidden=256,
        trunk_fourier_freq=2.0,
        trunk_emb_dim=256,

        # Common args
        basis_dim=256,
        out_dim=1,
        coord_dim=3,              # x, y, t
        act="tanh",
    )

    use_causality = True
    max_grad_norm = 1.0
    optimizer_name = "adam" # adam. soap
    alpha_w = 1.0 # moving average weight for loss balancing
    use_multi_gpu = True

    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=100,
        step_size=10.0,
        min_mean_weight=0.4,
        max_min_weight=0.99,
    )

    # training hyperparameters
    num_epochs = 15000
    initial_lr = 5e-4
    decay_every = 100
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 500
    resample_every = 10
    warmup_epochs = 500

    num_u_samples = 16
    num_pde_samples = 2048
    num_rar_samples = 0
    num_rar_pools = 0 # too slow to compute huge pool prediction, and no apparent benefit
    num_ic_samples = 1024

    lbd = 5.0
    N_val = 63  # element num in fem, points is N_val + 1
    M_val = 0.1

    @property
    def h_val(self):
        return 100 / self.N_val

    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
