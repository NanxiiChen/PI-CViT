from dataclasses import dataclass
import jax
import jax.numpy as jnp


class Config:
    """
    wave FNO 的 include_t0 + AD 协议:
    - 模型输出 T+1 帧, 时间标签覆盖 [0,1] 含 t=0 —— 单一时间网格
    - 两个初值条件都在帧 0 上精确施加(与 PDE 残差同一次查询):
        loss_ic      : u(x,0) == u0          (值锚, 精确)
        loss_ic_ut   : u_t(x,0) == 0          (用平移一阶 AD 导数, 无差分截断)
    - PDE 残差用 AD 二阶时间导数(平移方向嵌套 jvp), 空间仍用 FFT 精确
    - 运行: python -m wave.train_pifno --configs train_fno_ad --set save_name=xxx
    - 与 train_fno(fd, 旧 T 帧模型)的差异: 多输出 t=0 帧 + 时间导数换 AD +
      初速度由差分改 AD; fd 分支不做 include_t0 配套(参 burgers 的 TODO 说明)
    """
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
    active_loss_names = ("pde", "ic", "ic_ut")
    use_multi_gpu = False
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
        padding=(10, 0, 0),  # padding for (t, x, y)
        include_t0=True,     # 输出 T+1 帧, 标签 0, 1/T, ..., 1
    )
    time_scheme = "ad"  # 平移方向嵌套 jvp 二阶时间导数
    # gradnorm 动态加权三项 (pde, ic, ic_ut); burgers include_t0 A/B 实测
    # gradnorm 优于固定权重(0.7267 vs 0.7496 同 epoch), 且初值项锁定后
    # 权重自动向 pde 倾斜
    use_gradnorm = True

    use_causality = True
    causality_params = dict(
        num_chunks=24,
        initial_eps=1e-2,
        max_eps=50.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )

    max_grad_norm = 1.0
    optimizer_name = "soap"

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
