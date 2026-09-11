from dataclasses import dataclass
import jax.numpy as jnp


class Config:
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
    use_multi_gpu = True
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
        padding=(10, 0, 0) # padding for (t, x, y)
    )
    time_scheme = "rk4" # rk4 | fd | ad (ad: 空间 FFT + 时间导数用 AD 对角 Jacobian)
    # time_scheme="ad" 且 ad_derivative="diagonal" 的时间配点子采样间隔:
    # 每帧 AD 导数需要一次切向量传播(成本 ~ 一次模型前向), stride=1 全 100 帧
    # 实测 ~74s/步; shift 模式忽略此参数(单次 jvp, 全帧残差)
    ad_frame_stride = 5
    # "ad" 的配点选取方式: "evenly" 等间隔(每步相同); "random" 分层随机
    # (时间轴 1D LHS, 每步用新 key 重采样, 对称于坐标协议的 lhs 配点)
    ad_time_sampling = "random"
    # "ad" 的时间导数定义: "shift" 网格整体平移方向导数(单次 ones 切向量
    # jvp, ~1s/步, 全帧残差, 网格模型实现时间演化的自然机制, 默认);
    # "diagonal" 逐帧对角 Jacobian(每帧一次 jvp, ~16s/步@stride5, 需配点
    # 子采样), 二者对平移协变的神经场等价, diagonal 仅作对照
    ad_derivative = "shift"
    # "ad" 的第一步积分锚权重: u(1/T) ≈ u0 + (dt/2)(rhs(u0)+rhs(u(1/T))),
    # 把给定初值与模型第一帧焊接(rk4 defect 链第一环的对应物), 与残差同属
    # PDE 缺陷族, 固定权重即可; 缺此项时纯残差塌缩到平凡解。
    # 标定(400ep A/B, shift 导数): 初始"梯度平衡"权重 1.6e-5 会令 L2 冻结在
    # ~0.72(锚太弱); w∈[1,100] 表现相当(最低 0.624@w=1), 取 1.0
    ad_anchor_weight = 100.0
    # gradnorm 双项加权实测(200ep)与梯度平衡权重同病(L2 卡 ~0.73): 初始锚
    # 梯度独大 ~6e4 倍, 均衡化等于削弱锚 —— 该协议需要锚在早期占主导,
    # 故默认走固定权重 (1, ad_anchor_weight); 双项结构与 last_weights 机制
    # 保留, 想试 gradnorm 时设 True
    use_gradnorm = False
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
    save_every = 1000
    log_every = 5
    test_every = 25
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16
    
    nu = 0.01