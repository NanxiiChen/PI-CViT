from dataclasses import dataclass
import jax.numpy as jnp


class Config:
    model_name = "fno"
    data_dir = "./data/wave"
    save_dir = "/root/autodl-tmp/tf-logs/wave/fno"
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
    active_loss_names = ("pde", "ic_ut")
    use_multi_gpu = True
    Lc = 1.0
    Tc = 1.0
    
    model_params = dict(
        in_channels=1,
        out_channels=1,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=64,
        depth=4,
        activation="gelu",
        add_coords=True,
        padding=(10, 0, 0) # padding for (t, x, y)
    )
    # "fd": 时间二阶中心差分(现状); "ad": 平移方向 AD 二阶导(嵌套 jvp, 空间
    # 仍用 FFT 精确求导), 对应审稿人要求的 grid 协议 AD 消融;
    # IC 锚定由 loss_ic_ut 承担(u_t=0 的一阶差分同时把首帧焊在 u0 上)。
    # include_t0 + AD 的协议见 configs/train_fno_ad.py(该协议下初值与初速度
    # 都在同一网格的帧 0 上精确施加); TODO: fd 分支尚未适配 include_t0 模型
    time_scheme = "fd" # "fd" | "ad" ("rk4" 为历史别名, 等同 fd)
    # 固定权重 (pde, ic_ut) —— use_gradnorm=False 时生效; 设为 None 用 (1,1)
    weight_coef = [1.0, 20.0]
    use_causality = True
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
        max_eps=50.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )
    
    num_epochs = 40000
    initial_lr = 5e-4
    decay_every = 500
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 500
    log_every = 50
    test_every = 100
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16
    
    nu = 0.01