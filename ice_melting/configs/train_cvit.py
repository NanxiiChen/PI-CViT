from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    model_name = "cvit"
    data_dir = "./data/ice_melting/ellipse"
    save_dir = "/root/autodl-tmp/tf-logs/ice_melting/ellipse/cvit"
    target_ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    spatial_domain = ((-0.5, 0.5), (-0.5, 0.5))  # x range, y range, normalized
    temporal_domain = (0.0, 1.0)  # t range
    a_range = (20, 40)
    b_range = (20, 40)
    theta_range = (0.0, jnp.pi)
    spatial_domain_phys = ((-50.0, 50.0), (-50.0, 50.0))  # physical spatial domain
    Lc = 100.0  # xc = x / Lc
    Tc = 3.0  # tc = t / Tc

    # model hyperparameters
    model_params = dict(
        ## model:encoder
        patch_size=(8, 8),
        grid_size=(64, 64),
        in_channels=1,  # phi
        emb_dim=256,  # emb_dim for encoder
        depth=2,
        num_heads=8,
        ## model:decoder
        fourier_freq=2.0,
        dec_depth=2,
        dec_num_heads=8,
        dec_emb_dim=256,  # two times of ffe hidden dim (sin, cos)
        dec_mlp_act="tanh",
        num_mlp_layers=3,
        out_dim=1,  # phi,
        layer_norm_eps=1e-5,
    )

    use_causality = True
    max_grad_norm = 1.0

    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=100,
        step_size=10.0,
        min_mean_weight=0.4,
        max_min_weight=0.99,
    )

    # training hyperparameters
    num_epochs = 50000
    initial_lr = 5e-4
    decay_every = 100
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 500
    resample_every = 1

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
