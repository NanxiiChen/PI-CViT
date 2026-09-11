from dataclasses import dataclass
import jax
import jax.numpy as jnp


class Config:
    model_name = "fno"
    data_dir = "./data/ice_melting/ellipse_refined_dt"
    save_dir = "/root/autodl-tmp/tf-logs/ice_melting/ellipse/fno"
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
    Nx = 64
    Ny = 64

    # model hyperparameters
    model_params = dict(
        in_channels=1,
        out_channels=1,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=32,
        depth=4,
        activation="gelu",
        add_coords=True,
        padding=(10, 0, 0) # padding for (t, x, y)
    )
    # add_coords=True 引入 (t,x,y) 坐标输入通道: t 通道是 time_scheme="ad"
    # 的 AD 时间导数的求导入口(对时间坐标 Jacobian 的对角元); 注意这改变了
    # lifting 输入通道数(1+3), 与旧 add_coords=False 的 checkpoint 不兼容
    time_scheme = "rk4" # rk4 | fd | ad (ad: 空间用 cfg 的 spatial_scheme + 时间导数用 AD 对角 Jacobian)
    spatial_scheme = "fd"
    # time_scheme="ad" 且 ad_derivative="diagonal" 的时间配点子采样间隔;
    # shift 模式忽略此参数(单次 jvp, 全帧残差)
    ad_frame_stride = 5
    # "ad" 的配点选取方式: "evenly" 等间隔(每步相同); "random" 分层随机
    # (时间轴 1D LHS, 每步用新 key 重采样, 对称于坐标协议的 lhs 配点);
    # 仅 diagonal 模式生效
    ad_time_sampling = "random"
    # "ad" 的时间导数定义: "shift" 网格整体平移方向导数(单次 ones 切向量
    # jvp, 全帧残差, 网格模型实现时间演化的自然机制, 默认); "diagonal"
    # 逐帧对角 Jacobian(每帧一次 jvp, 需配点子采样), 二者对平移协变的
    # 神经场等价, diagonal 仅作对照
    ad_derivative = "shift"
    # "ad" 的第一步积分锚权重: phi(1/T) ≈ u0 + (dt/2)(rhs(u0)+rhs(phi(1/T))),
    # 把给定初值与模型第一帧焊接(rk4 defect 链第一环的对应物), 与残差同属
    # PDE 缺陷族, 固定权重即可; 缺此项时纯残差塌缩到平凡解(phi≡±1)。
    # 标定: 初始处锚/残差梯度范数 0.91/0.54, w=1.0 本身已接近梯度平衡
    ad_anchor_weight = 1.0
    # gradnorm 双项加权在 burgers 上实测与"梯度平衡权重"同病(削弱锚);
    # 固定权重 (1, ad_anchor_weight) 为默认, 想试 gradnorm 时设 True
    use_gradnorm = False

    use_causality = True
    max_grad_norm = 1.0
    optimizer_name = "adam" # adam. soap
    alpha_w = 1.0 # moving average weight for loss balancing

    causality_params = dict(
        num_chunks=10,
        initial_eps=1e-2,
        max_eps=100,
        step_size=5.0,
        min_mean_weight=0.4,
        max_min_weight=0.99,
    )

    # training hyperparameters
    num_epochs = 20000
    initial_lr = 5e-4
    decay_every = 100
    decay_rate = 0.95
    min_lr = 1e-6
    save_every = 500
    log_every = 5
    test_every = 25
    resample_u_every = 25
    warmup_epochs = 0
    num_u_samples = 16

    lbd = 5.0
    N_val = 63  # element num in fem, points is N_val + 1
    M_val = 0.1

    @property
    def h_val(self):
        return 100 / self.N_val

    @property
    def epsilon(self):
        return 6 * self.h_val / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
