from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from models.causal import CausalWeightor
from models.fno import FNO

# 逐帧 one-hot JVP 每批并行的帧数上限（time_scheme="ad"）。每次 JVP 的计算图
# 用 checkpoint 重计算，峰值显存 ~ B * chunk * 单次 JVP 中间激活量；默认 1 最
# 保守（生产配置 B=16, T=100, width=32, depth=8 在 96GB GPU 上实测安全），
# batch 较小、显存富余时可调大以提高并行度
AD_FRAME_CHUNK = 1
# ad 分支残差缩放系数。保持 1.0：导数本身经 float64 对拍验证精确，残差过小
# 的原因不是舍入而是缺 IC 锚定导致的平凡解塌缩（见 loss_pde 的 ad 分支）；
# 若非 1.0，注意 ad_ic_weight 的有效权重会被缩放平方倍（1e3 -> 1e6）压制
AD_PDE_SCALE = 1000.0


def _fft_k(n: int, length, dtype=jnp.float32):
    # cycles -> radians
    return (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=length / n).astype(dtype)


def _spectral_grad_lap_2d(u: jnp.ndarray, lx, ly):
    """
    u: (..., Nx, Ny)  # 与模型一致: ij / (x,y)
    return: ux, uy, lap with same shape
    """
    nx, ny = u.shape[-2], u.shape[-1]
    u_hat = jnp.fft.fftn(u, axes=(-2, -1))

    # fno中采用 nx, ny 顺序约定（区别于cvit/transformer中常见的 Nx, Ny 顺序），
    # 因此 kx 对应 axis=-2，ky 对应 axis=-1
    kx = _fft_k(nx, lx, dtype=u.dtype)  # (Nx,)
    ky = _fft_k(ny, ly, dtype=u.dtype)  # (Ny,)

    ikx = 1j * kx[:, None]              # (Nx, 1) 作用在 axis=-2
    iky = 1j * ky[None, :]              # (1, Ny) 作用在 axis=-1
    lap_mul = -(kx[:, None] ** 2 + ky[None, :] ** 2)  # (Nx, Ny)

    ux = jnp.fft.ifftn(u_hat * ikx, axes=(-2, -1)).real
    uy = jnp.fft.ifftn(u_hat * iky, axes=(-2, -1)).real
    lap = jnp.fft.ifftn(u_hat * lap_mul, axes=(-2, -1)).real
    return ux, uy, lap


def _time_derivative_fd(x: jnp.ndarray, dt):
    """
    x: (B, T, Nx, Ny)
    """
    xt = jnp.zeros_like(x)
    xt = xt.at[:, 1:-1].set((x[:, 2:] - x[:, :-2]) / (2.0 * dt))
    xt = xt.at[:, 0].set((x[:, 1] - x[:, 0]) / dt)
    xt = xt.at[:, -1].set((x[:, -1] - x[:, -2]) / dt)
    return xt


def _ad_collocation_idx(num_t: int, stride: int, key=None):
    # AD 时间配点帧下标（0-based，对应 pred 帧轴），stride=1 时为全部帧。
    # key=None: 等间隔（linspace，含首末帧，每步相同）；
    # key!=None: 分层随机——把 T 帧分成 K=ceil(T/stride) 个等宽段、每段内均匀
    # 取一帧（时间轴上的 1D LHS，与坐标协议 CoordSampler 的 lhs_sampling
    # 配点思想对称），保证每步覆盖整个时间轴（causal 每块都有配点），
    # 由训练循环每步传入新 key 实现逐步重采样
    num_k = -(-num_t // stride)  # ceil
    if key is None:
        return jnp.round(jnp.linspace(0, num_t - 1, num_k)).astype(jnp.int32)
    lows = jnp.arange(num_k) * stride
    offs = jax.random.randint(key, (num_k,), 0, stride)
    # 末段可能不足 stride 宽，clip 到 T-1（只影响末段，不会产生重复帧）
    return jnp.sort(jnp.minimum(lows + offs, num_t - 1))


def _time_derivative_ad(
    model: FNO, u0: jnp.ndarray, num_t: int, stride: int = 1, key=None
):
    """
    FNO 输出的 AD 时间导数（时间坐标 Jacobian 的对角元）。

    FNO 的时间坐标以输入坐标通道进入网络（_grid_3d 的 t 通道），但 3D 谱卷积
    沿时间耦合所有帧：输出第 k 帧依赖整个时间坐标向量 (t_1..t_T)，因此
    d(out)/d(t_coords) 是 (输出帧 x 标签帧) 的块 Jacobian。PDE 残差需要的对
    角元 d out_k / d t_k 通过 one-hot 切向量的 jvp 提取：对每个帧 k 只扰动第
    k 帧的时间标签，读取第 k 帧输出的切向量。语义上即把网络视为神经场
    u_theta(x, t) 后逐帧采样的 du/dtau（归一化时间 tau），与坐标协议中
    jacrev(decoder, argnums=t) 的 AD 时间导数对称。

    每帧一次切向量传播（成本 ~ 一次模型前向，FFT/卷积的切量运算是同尺寸的），
    实测全 100 帧的 loss+grad ~74s/步（rk4 基线 0.33s）；stride 对时间帧做
    配点子采样（见 _ad_collocation_idx，key 非空时为分层随机），成本/显存
    随帧数线性下降。

    u0: (C, Nx, Ny)
    return: (primal, ut)
        primal: (C, T, Nx, Ny)  模型输出帧 1..T（默认 t_coords 前向）
        ut:     (C, K, Nx, Ny)  K = ceil(T/stride) 个配点帧的 du/dtau，
                                对应帧下标 _ad_collocation_idx(T, stride, key)
    """
    dtype = u0.dtype
    t_base = jnp.arange(1, num_t + 1, dtype=dtype) / jnp.asarray(num_t, dtype=dtype)

    idx = _ad_collocation_idx(num_t, stride, key)
    num_k = idx.shape[0]
    eyes = jnp.eye(num_t, dtype=dtype)[idx]  # (K, T)，第 m 行 = e_{idx[m]}

    def fwd(tv):
        return model(u0, t_coords=tv)

    def one_hot_jvp(tan):
        # Jacobian 的第 idx[m] 列: tang[c, j] = d out_j / d t_{idx[m]}。
        # checkpoint: 外层对参数的 vjp 不保留全部 K 次 JVP 的中间激活（否则
        # 峰值显存 ~ B*K*单次 JVP 激活，实测 OOM），反传时按帧重计算；只返回
        # 切向量，避免存 K 份 primal
        def _jvp_tan(tan_):
            _, tang = jax.jvp(fwd, (t_base,), (tan_,))
            return tang

        return jax.checkpoint(_jvp_tan)(tan)  # (C, T, Nx, Ny)

    chunk = next(
        c for c in range(min(AD_FRAME_CHUNK, num_k), 0, -1) if num_k % c == 0
    )
    tangents = jax.lax.map(one_hot_jvp, eyes, batch_size=chunk)  # (K, C, T, Nx, Ny)
    # 对角提取: ut[c, m] = tangents[m, c, idx[m]] = d out_{idx[m]} / d t_{idx[m]}
    ut = tangents[jnp.arange(num_k), :, idx]  # (K, C, Nx, Ny)
    ut = jnp.moveaxis(ut, 0, 1)  # (C, K, Nx, Ny)

    primal = model(u0)  # 默认 t_coords 与 t_base 一致
    return primal, ut


def _time_derivative_shift(model: FNO, u0: jnp.ndarray, num_t: int):
    """
    FNO 输出的平移时间导数（网格整体平移方向的方向导数）:

        u_t = d/dδ out(u0, t_grid + δ*1) |_{δ=0}

    用 ones 切向量的单次 jvp 计算。对理想神经场（输出只依赖各帧标签值、
    平移协变）严格等于 du_theta/dt 在各帧的采样；FNO 的 3D 谱卷积沿时间
    耦合所有帧，"网络表达时间演化"的机制正是网格平移，因此平移导数是比
    逐帧对角导数更自洽的 u_t 定义，且成本 = 单次切向量传播（对角为 K 次，
    便宜 ad_frame_stride 倍，无需子采样）。

    u0: (C, Nx, Ny)
    return: (primal, ut)
        include_t0 模型: (C, T+1, Nx, Ny), ut[c, k] = 平移导数在帧 k
        (标签 k/T, k=0..T); 旧模型: (C, T, Nx, Ny), 帧 k 标签 (k+1)/T
    """
    dtype = u0.dtype
    if bool(getattr(model, "include_t0", False)):
        # include_t0: T+1 帧覆盖 [0,1] —— 单一网格, IC 与残差同查询
        t_base = (
            jnp.arange(0, num_t + 1, dtype=dtype) / jnp.asarray(num_t, dtype=dtype)
        )
    else:
        t_base = (
            jnp.arange(1, num_t + 1, dtype=dtype) / jnp.asarray(num_t, dtype=dtype)
        )

    def fwd(tv):
        return model(u0, t_coords=tv)

    _, ut = jax.jvp(
        jax.checkpoint(fwd), (t_base,), (jnp.ones_like(t_base),)
    )
    primal = model(u0)
    return primal, ut


def _burgers_rhs_normalized_time(
    state: jnp.ndarray, nu, lx, ly, lc, tc
):
    """
    state: (..., 2, Nx, Ny), normalized time tau in [0,1]
    PDE:
      u_t/tc + (u·∇)u/lc - nu*Δu/lc^2 = 0
    => du/dtau = tc * ( - (u·∇)u/lc + nu*Δu/lc^2 )
    """
    u = state[..., 0, :, :]
    v = state[..., 1, :, :]

    ux, uy, lap_u = _spectral_grad_lap_2d(u, lx=lx, ly=ly)
    vx, vy, lap_v = _spectral_grad_lap_2d(v, lx=lx, ly=ly)

    adv_u = u * ux + v * uy
    adv_v = u * vx + v * vy

    rhs_u = tc * (-adv_u / lc + nu * lap_u / (lc**2))
    rhs_v = tc * (-adv_v / lc + nu * lap_v / (lc**2))
    return jnp.stack([rhs_u, rhs_v], axis=-3)  # (..., 2, Nx, Ny)


class Losses(eqx.Module):
    causal_weightor: CausalWeightor | None = eqx.field(static=True, default=None)
    time_scheme: str = eqx.field(static=True, default="rk4")  # "rk4" | "fd" | "ad"

    def __init__(
        self,
        causal_weightor: CausalWeightor | None = None,
        time_scheme: str = "rk4",
    ):
        if time_scheme not in ("rk4", "fd", "ad"):
            raise ValueError("time_scheme must be 'rk4', 'fd' or 'ad'")
        self.causal_weightor = causal_weightor
        self.time_scheme = time_scheme

    @eqx.filter_jit
    def loss_pde(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, C, Nx, Ny), 对应 t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        if self.time_scheme == "ad":
            # ad 分支：前向与逐帧对角 JVP 在同一次 vmap 里完成；
            # ad_frame_stride 对时间帧做配点子采样（成本随帧数线性，见
            # _time_derivative_ad 的 docstring）；ad_time_sampling="random"
            # 时用训练循环传入的 colloc_key 做分层随机配点并逐步重采样
            stride = int(getattr(cfg, "ad_frame_stride", 1))
            sampling = str(getattr(cfg, "ad_time_sampling", "evenly"))
            if sampling not in ("evenly", "random"):
                raise ValueError("ad_time_sampling must be 'evenly' or 'random'")
            colloc_key = kwargs.get("colloc_key", None)
            if sampling == "random" and colloc_key is None:
                raise ValueError(
                    "ad_time_sampling='random' requires 'colloc_key' "
                    "(passed from the training loop each step)"
                )
            idx_key = colloc_key if sampling == "random" else None
            # u_t 的 AD 方向: "shift" 网格整体平移方向(Jacobian-向量积 J·1,
            # 单次 jvp, 网格模型实现时间演化的自然机制, 默认); "diagonal" 逐帧
            # 对角元(每帧一次 one-hot jvp, 需配点子采样控成本, 帧局部理想化)
            deriv_mode = str(getattr(cfg, "ad_derivative", "shift"))
            include_t0 = bool(getattr(model, "include_t0", False))
            if deriv_mode == "shift":
                # 平移导数: 单次 ones 切向量 jvp, 全帧残差, 无需配点子采样;
                # include_t0 模型输出 T+1 帧(标签 0..1), 残差只取未来帧 [1:]
                pred, uts = jax.vmap(
                    _time_derivative_shift, in_axes=(None, 0, None)
                )(model, u0, model.time_steps)
                frame_idx = (
                    jnp.arange(1, model.time_steps + 1)
                    if include_t0
                    else jnp.arange(model.time_steps)
                )
            elif deriv_mode == "diagonal":
                if include_t0:
                    raise ValueError(
                        "ad_derivative='diagonal' 未适配 include_t0 模型, 请用 'shift'"
                    )
                pred, uts = jax.vmap(
                    _time_derivative_ad, in_axes=(None, 0, None, None, None)
                )(model, u0, model.time_steps, stride, idx_key)
                frame_idx = _ad_collocation_idx(model.time_steps, stride, idx_key)  # (K,)
            else:
                raise ValueError("ad_derivative must be 'diagonal' or 'shift'")
            # pred: (B, C, T(+1), Nx, Ny); uts: (B, C, K, Nx, Ny)
            # 注: 初值锚定 —— include_t0 模型由 loss_ic 直接约束输出帧 0
            # (同一次查询, 无第二套坐标网格); 旧模型由 loss_anchor 的第一步
            # trapezoid 积分锚带入
        else:
            pred = jax.vmap(model)(u0)  # (B, C, T, Nx, Ny), 统一约定：对应 t1..tT
            uts = None
        if pred.ndim != 5:
            raise ValueError(f"FNO output must be (B,C,T,Nx,Ny), got {pred.shape}")
        if pred.shape[1] != 2:
            raise ValueError(f"Burgers expects C=2, got C={pred.shape[1]}")

        b, c, t, Nx, Ny = pred.shape
        if t < 1:
            raise ValueError("pred must contain at least one future step")

        # 统一约定：显式拼接 t=0 初值，形成完整时间序列 [t0, t1, ..., tT]
        # （fd/rk4 在完整序列上定义；ad 不使用初值帧——初值无模型时间导数；
        #  TODO: fd/rk4 分支尚未适配 include_t0 模型, 见下方守卫）
        if self.time_scheme in ("fd", "rk4") and bool(
            getattr(model, "include_t0", False)
        ):
            raise NotImplementedError(
                "TODO: fd/rk4 分支尚未适配 include_t0 模型(需处理 T+1 帧序列"
                "与 dt/ts 约定); 当前仅 time_scheme='ad' 支持 include_t0"
            )
        pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,C,T+1,Nx,Ny)

        nu = getattr(cfg, "nu", 0.01)
        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)

        # 统一时间步长：pred 有 T 帧未来时间 => dt = 1/T
        dt = 1.0 / t

        if self.time_scheme == "ad":
            # u_t 来自网络对时间坐标输入的 AD 导数（对角 Jacobian），只在 K 个
            # 配点帧上求残差；空间项与 fd 分支一致用精确的谱导数
            u = pred[:, 0][:, frame_idx]  # (B, K, Nx, Ny)
            v = pred[:, 1][:, frame_idx]
            ut = uts[:, 0][:, frame_idx]  # (B, K, Nx, Ny), du/dtau
            vt = uts[:, 1][:, frame_idx]

            ux, uy, lap_u = _spectral_grad_lap_2d(u, lx=lx, ly=ly)
            vx, vy, lap_v = _spectral_grad_lap_2d(v, lx=lx, ly=ly)

            ru = ut / tc + (u * ux + v * uy) / lc - nu * lap_u / (lc**2)
            rv = vt / tc + (u * vx + v * vy) / lc - nu * lap_v / (lc**2)
            
            ru = ru * AD_PDE_SCALE
            rv = rv * AD_PDE_SCALE

            res_sq_t = jnp.mean(ru ** 2 + rv ** 2, axis=(-2, -1))  # (B, K)
            # 帧时间标签: include_t0 时帧 k 即 t=k/T(frame_idx 已含 +1);
            # 旧约定帧 j 为 t=(j+1)/T
            if bool(getattr(model, "include_t0", False)):
                ts = (frame_idx.astype(pred.dtype) * dt).reshape(-1, 1)
            else:
                ts = ((frame_idx + 1).astype(pred.dtype) * dt).reshape(-1, 1)
            # 注: 初值锚定在独立 loss 里与残差双项加权(见 loss_fn):
            # include_t0 模型用 loss_ic(帧 0 == u0), 旧模型用 loss_anchor

        elif self.time_scheme == "fd":
            u = pred_full[:, 0, :, :, :]  # (B,T+1,Nx,Ny)
            v = pred_full[:, 1, :, :, :]

            ut = _time_derivative_fd(u, dt)
            vt = _time_derivative_fd(v, dt)

            ux, uy, lap_u = _spectral_grad_lap_2d(u, lx=lx, ly=ly)
            vx, vy, lap_v = _spectral_grad_lap_2d(v, lx=lx, ly=ly)

            ru = ut / tc + (u * ux + v * uy) / lc - nu * lap_u / (lc**2)
            rv = vt / tc + (u * vx + v * vy) / lc - nu * lap_v / (lc**2)

            res_sq_t = jnp.mean(ru[:, 1:] ** 2 + rv[:, 1:] ** 2, axis=(-2, -1))  # (B,T)
            ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        else:  # rk4
            states = jnp.swapaxes(pred_full, 1, 2)  # (B,T+1,C,Nx,Ny)
            s_n = states[:, :-1, :, :, :]           # (B,T,C,Nx,Ny) 对应 0..T-1
            s_np1 = states[:, 1:, :, :, :]          # (B,T,C,Nx,Ny) 对应 1..T

            rhs = lambda s: _burgers_rhs_normalized_time(
                s, nu=nu, lx=lx, ly=ly, lc=lc, tc=tc
            )

            k1 = rhs(s_n)
            k2 = rhs(s_n + 0.5 * dt * k1)
            k3 = rhs(s_n + 0.5 * dt * k2)
            k4 = rhs(s_n + dt * k3)

            s_rk4 = s_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            defect = s_np1 - s_rk4  # (B,T,C,Nx,Ny)

            res_sq_t = jnp.mean(defect[:, :, 0] ** 2 + defect[:, :, 1] ** 2, axis=(-2, -1))  # (B,T)
            ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        # -------------------------
        # causal / non-causal
        # -------------------------
        if bool(getattr(cfg, "use_causality", False)) and self.causal_weightor is not None:
            eps = kwargs.get("causal_eps", None)
            residuals_for_causal = jnp.sqrt(jnp.clip(res_sq_t, min=1e-30))
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            return loss, {
                "loss_chunks": loss_chunks,
                "causal_weights": causal_weights,
                "residual_t_mean": jnp.mean(res_sq_t, axis=0),
            }

        loss = jnp.mean(res_sq_t)
        return loss, {"residual_t_mean": jnp.mean(res_sq_t, axis=0)}

    @eqx.filter_jit
    def loss_anchor(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, C, Nx, Ny), 对应 t=0
        cfg: Any,
        **kwargs,
    ):
        # ad 专用第一步积分锚 (trapezoid): u(1/T) ≈ u0 + (dt/2)(rhs(u0)+rhs(u(1/T)))
        # 把给定初值 u0 与模型第一帧焊接, 起 rk4 defect 链第一环的作用; 与残差
        # 同在标准网格查询内(不引入第二套时间网格), rhs 空间导数用谱方法直接
        # 算, 无需对网络求导。作为独立 loss 与 loss_pde 走 gradnorm 加权
        if bool(getattr(model, "include_t0", False)):
            raise ValueError(
                "loss_anchor 假定帧 0 = t=1/T; include_t0 模型的帧 0 是 t=0, "
                "请用 loss_ic"
            )
        nu = getattr(cfg, "nu", 0.01)
        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        dt = 1.0 / model.time_steps

        pred = jax.vmap(model)(u0)  # 默认网格 (1/T, ..., 1)
        rhs0 = _burgers_rhs_normalized_time(u0, nu=nu, lx=lx, ly=ly, lc=lc, tc=tc)
        rhs1 = _burgers_rhs_normalized_time(
            pred[:, :, 0], nu=nu, lx=lx, ly=ly, lc=lc, tc=tc
        )
        anchor = pred[:, :, 0] - u0 - 0.5 * dt * (rhs0 + rhs1)  # (B,C,Nx,Ny)
        return jnp.mean(anchor ** 2), {}

    @eqx.filter_jit
    def loss_ic(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, C, Nx, Ny), 对应 t=0
        cfg: Any,
        **kwargs,
    ):
        # ad + include_t0 专用值锚: 模型输出帧 0 即 u_theta(x, t=0), 与 PDE
        # 残差同一次查询(单一时间网格覆盖 [0,1]), 直接约束其等于给定初值 ——
        # 相比 loss_anchor 的 trapezoid 缺陷(锚在 O(dt^3) 松弛上)这是精确的
        # 值锚定, 且彻底不存在第二套坐标网格(mode 分离的根源)
        if not bool(getattr(model, "include_t0", False)):
            raise ValueError(
                "loss_ic 要求 FNO(include_t0=True): 输出帧 0 必须对应 t=0 "
                "(旧模型的帧 0 是 t=1/T, 请用 loss_anchor)"
            )
        pred = jax.vmap(model)(u0)
        return jnp.mean((pred[:, :, 0] - u0) ** 2), {}

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, C, Nx, Ny)
        cfg: Any,
        last_weights: jnp.ndarray | None = None,
        **kwargs,
    ):
        if self.time_scheme == "ad":
            # 双项 (loss_pde, 锚项) + gradnorm/固定加权, 镜像坐标协议 losses.py;
            # 锚项按模型约定选取: include_t0 -> loss_ic (帧 0 == u0, 精确值锚,
            # 单一网格); 旧模型 -> loss_anchor (第一步 trapezoid 积分锚)
            anchor_term = (
                "loss_ic"
                if bool(getattr(model, "include_t0", False))
                else "loss_anchor"
            )
            losses = []
            grads = []
            aux_vars = {}
            for name in ("loss_pde", anchor_term):
                vg_fn = eqx.filter_value_and_grad(getattr(self, name), has_aux=True)
                (loss, aux), grad = vg_fn(model, u0=u0, cfg=cfg, **kwargs)
                grad = jax.tree.map(lambda g: jnp.nan_to_num(g), grad)
                losses.append(loss)
                grads.append(grad)
                aux_vars.update({name: loss, **aux})

            if getattr(cfg, "use_gradnorm", True):
                weights = self.grad_norm_weights(grads)
            else:
                # 固定权重回退: (残差=1, 锚=ad_anchor_weight)
                weights = jnp.asarray(
                    [1.0, float(getattr(cfg, "ad_anchor_weight", 1.0))]
                )
            weight_coef = jnp.asarray(getattr(cfg, "ad_weight_coef", (1.0, 1.0)))
            alpha_w = float(getattr(cfg, "alpha_w", 1.0))
            if last_weights is None:
                last_weights = jnp.ones(len(losses))
            weights = weights * weight_coef
            weights = alpha_w * weights + (1 - alpha_w) * last_weights

            total_loss = jnp.sum(jnp.stack(losses) * weights)
            total_grad = jax.tree.map(
                lambda *gs: jnp.sum(
                    jnp.stack([w * g for w, g in zip(weights, gs)]), axis=0
                ),
                *grads,
            )
            aux_vars["weights"] = weights
            return (total_loss, aux_vars), total_grad

        # rk4 / fd: 单损失路径
        vg_fn = eqx.filter_value_and_grad(self.loss_pde, has_aux=True)
        (loss, aux), grad = vg_fn(model, u0=u0, cfg=cfg, **kwargs)
        return (loss, aux), grad

    @staticmethod
    def grad_norm_weights(grads: list, eps=1e-6):
        # 与坐标协议 losses.py 的 grad_norm_weights 相同
        def tree_norm(pytree):
            leaves = jax.tree.leaves(pytree)
            return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / grad_norms
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)


if __name__ == "__main__":
    import time as _time

    B, C, Nx, Ny = 2, 2, 16, 16
    T = 20
    key = jax.random.PRNGKey(0)
    model = FNO(
        key=key,
        in_channels=2,
        out_channels=2,
        time_steps=T,
        modes_t=4,
        modes_x=4,
        modes_y=4,
        width=8,
        depth=2,
        add_coords=True,
        padding=(2, 0, 0),
    )
    u0 = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (B, C, Nx, Ny))

    cfg = type("Config", (), {"nu": 0.01, "Lc": 1.0, "Tc": 1.0, "lx": 1.0, "ly": 1.0})

    # 1) 三种 time_scheme 的 loss + 梯度冒烟测试（计时含首次编译）；
    #    ad 默认走 shift 路径, 另显式覆盖 diagonal(含配点子采样与 causal
    #    weighting, 后者验证 K<K_full 时 residual/ts 接口兼容)
    causal_w = CausalWeightor(num_chunks=5, t_range=(0.0, 1.0))
    for scheme, extra, cw in (
        ("rk4", {}, None),
        ("fd", {}, None),
        ("ad", {}, None),  # shift 默认路径
        ("ad", {"ad_derivative": "diagonal"}, None),
        ("ad", {"ad_derivative": "diagonal", "ad_frame_stride": 2,
                "use_causality": True}, causal_w),
        ("ad", {"ad_derivative": "diagonal", "ad_frame_stride": 2,
                "ad_time_sampling": "random", "use_causality": True}, causal_w),
    ):
        cfg_i = type("Config", (), {
            **vars(cfg), "use_causality": cw is not None, **extra,
        })
        losses = Losses(causal_weightor=cw, time_scheme=scheme)
        t0 = _time.perf_counter()
        ck, sk = jax.random.split(jax.random.PRNGKey(123))
        (loss_value, aux), grad = losses.loss_fn(
            model, u0=u0, cfg=cfg_i, causal_eps=jnp.array(1e-2), colloc_key=ck
        )
        loss_value.block_until_ready()
        grads_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grad)
        )
        print(
            f"[{scheme} {extra or 'full'}] loss={float(loss_value):.4e}  "
            f"grad_finite={bool(grads_finite)}  "
            f"({_time.perf_counter() - t0:.1f}s incl. compile)"
        )

    # 2) AD 对角导数正确性: 与 "只扰动第 k 帧时间标签" 的中心差分对拍,
    #    应收敛到 O(delta^2) (float32 下差分自身的舍入放大到 ~1e-2, 无法区分
    #    实现误差, 故此处临时开 x64 复验); 另验证 stride 子采样取值一致
    pred, uts = jax.vmap(
        _time_derivative_ad, in_axes=(None, 0, None, None)
    )(model, u0, T, 1)
    pred_plain = jax.vmap(model)(u0)
    print(
        "primal(t_coords=t_base) == default forward:",
        bool(jnp.allclose(pred, pred_plain, atol=1e-6)),
    )

    _, uts_s2 = _time_derivative_ad(model, u0[0], T, 2)
    idx2 = _ad_collocation_idx(T, 2)
    # f32 下不同 lax.map 批次的 FFT 归约顺序有 ~1e-3 量级的相对抖动(良性),
    # 正确性由下面 float64 的对拍保证
    print(
        "stride=2 matches full at collocation frames:",
        bool(jnp.allclose(uts_s2, uts[0][:, idx2], rtol=1e-3, atol=1e-5)),
    )

    # 分层随机配点: 严格递增(无重复)、帧数正确, 且导数与全帧在所选帧处一致
    for sk in jax.random.split(jax.random.PRNGKey(7), 3):
        ridx = _ad_collocation_idx(T, 2, sk)
        assert ridx.shape == (10,) and bool(jnp.all(jnp.diff(ridx) > 0))
    _, uts_r2 = _time_derivative_ad(model, u0[0], T, 2, sk)
    print(
        "random collocation: strictly increasing K=10; "
        "matches full at sampled frames:",
        bool(jnp.allclose(uts_r2, uts[0][:, ridx], rtol=1e-3, atol=1e-5)),
    )

    jax.config.update("jax_enable_x64", True)
    model64 = jax.tree.map(
        lambda a: a.astype(jnp.float64) if isinstance(a, jax.Array) else a, model
    )
    u0_64 = u0.astype(jnp.float64)
    _, uts64 = jax.vmap(
        _time_derivative_ad, in_axes=(None, 0, None, None)
    )(model64, u0_64, T, 1)

    delta = 1e-3
    t_base = jnp.arange(1, T + 1, dtype=uts64.dtype) / T

    def label_perturb_fd(e_k):
        out_p = jax.vmap(model64, in_axes=(0, None))(u0_64, t_base + delta * e_k)
        out_m = jax.vmap(model64, in_axes=(0, None))(u0_64, t_base - delta * e_k)
        return (out_p - out_m) / (2.0 * delta)

    ut_fd = jax.vmap(label_perturb_fd)(jnp.eye(T, dtype=uts64.dtype))  # (T_k, B, C, T_j, Nx, Ny)
    ut_fd_diag = jnp.diagonal(ut_fd, axis1=0, axis2=3)  # (B, C, Nx, Ny, T_k)
    ut_fd_diag = jnp.moveaxis(ut_fd_diag, -1, 2)  # (B, C, T, Nx, Ny)
    rel = jnp.linalg.norm(uts64 - ut_fd_diag) / jnp.linalg.norm(ut_fd_diag)
    print(
        f"AD vs label-perturbation FD (float64, delta={delta:.0e}): "
        f"u_t relative diff = {float(rel):.3e}"
    )

    # shift 平移导数正确性: 与 "整体平移全部标签 ±delta" 的中心差分对拍,
    # 应同样收敛到 O(delta^2) —— 两种模式都是精确 AD, 只是方向不同
    _, uts64_shift = jax.vmap(
        _time_derivative_shift, in_axes=(None, 0, None)
    )(model64, u0_64, T)
    ones = jnp.ones(T, dtype=uts64.dtype)
    out_p = jax.vmap(model64, in_axes=(0, None))(u0_64, t_base + delta * ones)
    out_m = jax.vmap(model64, in_axes=(0, None))(u0_64, t_base - delta * ones)
    ut_fd_shift = (out_p - out_m) / (2.0 * delta)
    rel_shift = (
        jnp.linalg.norm(uts64_shift - ut_fd_shift) / jnp.linalg.norm(ut_fd_shift)
    )
    print(
        f"shift vs grid-shift FD (float64, delta={delta:.0e}): "
        f"u_t relative diff = {float(rel_shift):.3e}"
    )

    # 3) include_t0 协议: T+1 帧单一网格(标签 0..1), loss_ic 值锚与 shift AD
    #    残差同查询; 验证输出形状/默认网格等价/loss 项选择/梯度有限,
    #    以及平移导数在新标签约定下的对拍
    # x64 已开, 新构造模型的初始化器会给出 float64 权重 —— 与 f32 输入会
    # 在 conv 处 dtype 冲突, 统一 cast 回 u0 的 dtype
    model_t0 = jax.tree.map(
        lambda a: a.astype(u0.dtype) if isinstance(a, jax.Array) else a,
        FNO(
            key=key,
            in_channels=2,
            out_channels=2,
            time_steps=T,
            modes_t=4,
            modes_x=4,
            modes_y=4,
            width=8,
            depth=2,
            add_coords=True,
            padding=(2, 0, 0),
            include_t0=True,
        ),
    )
    out_t0 = model_t0(u0[0])
    t0_grid = jnp.arange(0, T + 1, dtype=u0.dtype) / T
    out_t0_explicit = model_t0(u0[0], t_coords=t0_grid)
    print(
        "include_t0: out shape (C,T+1,Nx,Ny):", tuple(out_t0.shape),
        " default == explicit t_coords:",
        bool(jnp.allclose(out_t0, out_t0_explicit, atol=1e-6)),
    )

    losses_t0 = Losses(time_scheme="ad")
    cfg_t0 = type("Config", (), {**vars(cfg), "use_causality": False})
    (l_t0, aux_t0), g_t0 = losses_t0.loss_fn(model_t0, u0=u0, cfg=cfg_t0)
    l_t0.block_until_ready()
    gf_t0 = jax.tree_util.tree_all(
        jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), g_t0)
    )
    has_ic = "loss_ic" in aux_t0 and "loss_anchor" not in aux_t0
    print(
        f"[ad include_t0] loss={float(l_t0):.4e}  grad_finite={bool(gf_t0)}  "
        f"anchor term = loss_ic: {has_ic}"
    )

    # 平移导数在新约定下与整体平移差分对拍 (x64 已开)
    model_t0_64 = jax.tree.map(
        lambda a: a.astype(jnp.float64) if isinstance(a, jax.Array) else a, model_t0
    )
    _, ut_t0_64 = _time_derivative_shift(model_t0_64, u0_64[0], T)
    t0_grid64 = jnp.arange(0, T + 1, dtype=jnp.float64) / T
    ones64 = jnp.ones(T + 1, dtype=jnp.float64)
    p = model_t0_64(u0_64[0], t_coords=t0_grid64 + delta * ones64)
    m = model_t0_64(u0_64[0], t_coords=t0_grid64 - delta * ones64)
    fd_t0 = (p - m) / (2.0 * delta)
    rel_t0 = jnp.linalg.norm(ut_t0_64 - fd_t0) / jnp.linalg.norm(fd_t0)
    print(
        f"include_t0 shift vs FD (float64, delta={delta:.0e}): "
        f"u_t relative diff = {float(rel_t0):.3e}"
    )