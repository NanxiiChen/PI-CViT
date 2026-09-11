from dataclasses import dataclass
import jax.numpy as jnp


class Config:
    """
    burgers FNO 的 include_t0 + AD 协议:
    - 模型输出 T+1 帧, 时间标签覆盖 [0,1] 含 t=0 —— 单一时间网格,
      loss_ic 直接用输出帧 0 对齐给定初值(精确值锚, 无第二套坐标网格,
      无 mode 分离), loss_pde 的 AD 平移导数也在同一网格上
    - 运行: python -m burgers.train_pifno --configs train_fno_ad --set save_name=xxx
    - 消融对照的 fd/rk4 臂沿用旧协议(模型输出 T 帧, train_fno config),
      include_t0 只用于 AD 臂; fd/rk4 分支不做 include_t0 配套
    """
    model_name = "fno"
    data_dir = "./data/burgers"
    save_dir = "/root/autodl-tmp/tf-logs/burgers/fno"
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
    use_multi_gpu = False
    Lc = 1.0
    Tc = 1.0

    model_params = dict(
        in_channels=2,
        out_channels=2,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=32,
        depth=8,
        activation="gelu",
        add_coords=True,
        padding=(10, 0, 0),  # padding for (t, x, y)
        include_t0=True,     # 输出 T+1 帧, 标签 0, 1/T, ..., 1
    )
    # AD 协议: 空间 FFT + 时间导数 = 网格平移方向导数(单次 jvp);
    # IC 锚 = loss_ic(同查询帧 0), 不需要也不适用 loss_anchor
    time_scheme = "ad"
    ad_derivative = "shift"
    ad_time_sampling = "random"   # 仅 diagonal 模式消耗
    ad_frame_stride = 5           # 仅 diagonal 模式消耗
    # 锚项(loss_ic)权重: gradnorm 动态加权(默认) —— include_t0 A/B 实测
    # (600ep): 固定 (1,100) 在 ~600ep 到 0.750 后缓慢回升, gradnorm 同期
    # 0.729 仍单调降且已越过固定臂最小值; IC 锁定后权重自动翻转到 pde 主导
    ad_anchor_weight = 100.0   # 仅 use_gradnorm=False 时生效
    use_gradnorm = True
    use_causality = True

    max_grad_norm = 1.0
    optimizer_name = "soap"
    causality_params = dict(
        num_chunks=10,
        initial_eps=1e-2,
        max_eps=5.0,
        step_size=5.0,
        min_mean_weight=0.2,
        max_min_weight=0.99,
    )

    num_epochs = 15000
    initial_lr = 5e-4
    decay_every = 200
    decay_rate = 0.95
    min_lr = 1e-5
    save_every = 2000
    log_every = 10
    test_every = 50
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16

    nu = 0.01
