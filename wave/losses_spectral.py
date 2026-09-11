from typing import Any, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from models.causal import CausalWeightor
from models.fno import FNO


def _fft_k(n: int, length, dtype=jnp.float32):
    # cycles -> radians
    return (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=length / n).astype(dtype)


def _spectral_lap_2d(u: jnp.ndarray, lx, ly):
    """
    u: (..., Nx, Ny), 约定 axis=-2 对应 x, axis=-1 对应 y
    return: lap with same shape
    """
    nx, ny = u.shape[-2], u.shape[-1]
    u_hat = jnp.fft.fftn(u, axes=(-2, -1))

    kx = _fft_k(nx, lx, dtype=u.dtype)  # (Nx,)
    ky = _fft_k(ny, ly, dtype=u.dtype)  # (Ny,)
    lap_mul = -(kx[:, None] ** 2 + ky[None, :] ** 2)  # (Nx, Ny)

    lap = jnp.fft.ifftn(u_hat * lap_mul, axes=(-2, -1)).real
    return lap


def _time_second_derivative_fd(x: jnp.ndarray, dt):
    """
    x: (B, T, Nx, Ny), 这里 T 是时间长度（例如 pred_full 的 T+1）
    返回 d2x/dtau2
    """
    n = x.shape[1]
    if n < 3:
        raise ValueError("Need at least 3 time points for second derivative.")

    h2 = dt**2
    xtt = jnp.zeros_like(x)

    # interior: 二阶中心差分
    xtt = xtt.at[:, 1:-1].set((x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / h2)

    if n >= 4:
        # boundary: 二阶单边差分（比原实现更合理）
        xtt = xtt.at[:, 0].set((2.0 * x[:, 0] - 5.0 * x[:, 1] + 4.0 * x[:, 2] - x[:, 3]) / h2)
        xtt = xtt.at[:, -1].set((2.0 * x[:, -1] - 5.0 * x[:, -2] + 4.0 * x[:, -3] - x[:, -4]) / h2)
    else:
        # n==3 退化情形
        xtt = xtt.at[:, 0].set((x[:, 2] - 2.0 * x[:, 1] + x[:, 0]) / h2)
        xtt = xtt.at[:, -1].set((x[:, -1] - 2.0 * x[:, -2] + x[:, -3]) / h2)

    return xtt


def _time_second_derivative_ad(model: FNO, u0: jnp.ndarray, num_t: int):
    """
    FNO 输出的 AD 二阶时间导数（平移方向的二阶方向导数）:

        u_tt = d^2/ddelta^2 out(u0, t_grid + delta*1) |_{delta=0}

    外层 jvp 给出平移一阶导数（对应 du/dtau，见 burgers 的 shift 模式），再套
    一层 jvp 得二阶；对平移协变的神经场，这等于 du^2/dtau^2 在各帧的采样。
    波动方程是时间二阶，PDE 残差需要 u_tt；空间项保持 FFT（周期问题精确）。

    成本: 约 2 次前向 + 2 次切向量传播（嵌套 jvp 需要二阶前向 AD）；
    checkpoint 控显存，参数梯度可穿过双重 jvp（float64 下与 shift 中心差分
    按 O(delta^2) 收敛，已验证）。

    u0: (1, Nx, Ny)
    return: (primal, utt)，均对应帧 1..T（标签 1/T..1）
    """
    dtype = u0.dtype
    t_base = jnp.arange(1, num_t + 1, dtype=dtype) / jnp.asarray(num_t, dtype=dtype)

    def g(delta):
        return jax.checkpoint(lambda d: model(u0, t_coords=t_base + d))(delta)

    def dg(delta):
        # g 对 delta 的一阶导（平移一阶导数的场）
        _, tang = jax.jvp(g, (delta,), (jnp.ones_like(delta),))
        return tang

    zero = jnp.zeros((), dtype)
    one = jnp.ones((), dtype)
    primal, _ = jax.jvp(g, (zero,), (one,))   # primal + 一阶导
    _, utt = jax.jvp(dg, (zero,), (one,))     # 二阶导
    return primal, utt


class Losses(eqx.Module):
    causal_weightor: CausalWeightor | None = eqx.field(static=True, default=None)
    time_scheme: str = eqx.field(static=True, default="fd")  # "fd" | "ad"

    def __init__(
        self,
        causal_weightor: CausalWeightor | None = None,
        time_scheme: str = "fd",
        **kwargs,
    ):
        # "rk4" 为历史遗留别名(该变量的值此前未被使用, 实际走的就是 fd 分支)
        if time_scheme == "rk4":
            time_scheme = "fd"
        if time_scheme not in ("fd", "ad"):
            raise ValueError("time_scheme must be 'fd' or 'ad' (or legacy 'rk4')")
        self.causal_weightor = causal_weightor
        self.time_scheme = time_scheme

    @eqx.filter_jit
    def loss_pde(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, 1, Nx, Ny), t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        if self.time_scheme == "ad":
            # AD 二阶时间导数(平移方向嵌套 jvp), 见 _time_second_derivative_ad;
            # 空间项保持 FFT(周期问题精确)。IC 锚定由 loss_ic_ut 承担(u_t=0
            # 的一阶差分同时把首帧焊在 u0 上), 故此处只算 PDE 残差
            if not model.add_coords:
                raise ValueError(
                    "time_scheme='ad' requires FNO(add_coords=True): "
                    "the time-coordinate input channel is the AD entry point"
                )
            pred, utt = jax.vmap(
                _time_second_derivative_ad, in_axes=(None, 0, None)
            )(model, u0, model.time_steps)  # (B,1,T,Nx,Ny), 帧 1..T
            if pred.ndim != 5:
                raise ValueError(f"FNO output must be (B,C,T,Nx,Ny), got {pred.shape}")
            if pred.shape[1] != 1:
                raise ValueError(f"Wave expects C=1, got C={pred.shape[1]}")
            b, c, t, nx, ny = pred.shape
            if t < 2:
                raise ValueError("Wave PDE needs at least 2 future steps (T>=2).")
            u_use = pred[:, 0]          # (B,T,Nx,Ny)
            utt_use = utt[:, 0]         # (B,T,Nx,Ny)
        else:
            pred = jax.vmap(model)(u0)  # (B, C, T, Nx, Ny), 对应 t1..tT
            if pred.ndim != 5:
                raise ValueError(f"FNO output must be (B,C,T,Nx,Ny), got {pred.shape}")
            if pred.shape[1] != 1:
                raise ValueError(f"Wave expects C=1, got C={pred.shape[1]}")
            b, c, t, nx, ny = pred.shape
            if t < 2:
                raise ValueError("Wave PDE needs at least 2 future steps (T>=2).")

            # [t0, t1, ..., tT]
            pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,1,T+1,Nx,Ny)
            u_full = pred_full[:, 0, :, :, :]  # (B,T+1,Nx,Ny)
            u_use, utt_use = u_full, None

        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)

        dt = 1.0 / t  # normalized tau step

        # c(x,y)
        x = jnp.linspace(0.0, lx, nx, endpoint=False, dtype=u_use.dtype)
        y = jnp.linspace(0.0, ly, ny, endpoint=False, dtype=u_use.dtype)
        xx, yy = jnp.meshgrid(x, y, indexing="ij")  # (Nx, Ny)
        c_xy = 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * xx / lx) * jnp.sin(2.0 * jnp.pi * yy / ly)
        c2 = c_xy**2  # (Nx, Ny)

        if self.time_scheme == "ad":
            lap = _spectral_lap_2d(u_use, lx=lx, ly=ly)
            # d2u/dt2 = (1/tc^2) d2u/dtau2
            residual = utt_use / (tc**2) - c2[None, None, :, :] * lap / (lc**2)
            res_sq_t = jnp.mean((residual / 100.0) ** 2, axis=(-2, -1))  # (B,T)
        else:
            utt = _time_second_derivative_fd(u_use, dt=dt)      # d2u/dtau2
            lap = _spectral_lap_2d(u_use, lx=lx, ly=ly)
            residual = utt / (tc**2) - c2[None, None, :, :] * lap / (lc**2)  # (B,T+1,Nx,Ny)
            residual = residual / 100.0  # scale for numerical stability
            # 仅统计未来时刻 t1..tT
            res_sq_t = jnp.mean(residual[:, 1:] ** 2, axis=(-2, -1))  # (B,T)

        ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        if bool(getattr(cfg, "use_causality", False)) and self.causal_weightor is not None:
            eps = kwargs.get("causal_eps", None)
            residuals_for_causal = jnp.sqrt(res_sq_t)
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            return loss, {
                "loss_chunks": loss_chunks,
                "causal_weights": causal_weights,
                "residual_t_mean": jnp.mean(res_sq_t, axis=0),
            }

        return jnp.mean(res_sq_t), {"residual_t_mean": jnp.mean(res_sq_t, axis=0)}

    @eqx.filter_jit
    def loss_ic_ut(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B,1,Nx,Ny), t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        # IC 锚定项, 同时承担两个角色:
        #   1) 物理初速度条件 u_t(0) = 0（波动方程的第二初值）
        #   2) 值锚定: 一阶差分 (u1-u0)/(dt*tc) 把首帧焊在给定初值 u0 上,
        #      等价于 burgers/ice 里显式的第一步积分锚 —— AD 残差本身与 u0
        #      解耦, 靠这一项排除平凡解(u≡0 被 u0≠0 排除, u≡const 被 Δu≠0
        #      排除), 故 time_scheme="ad" 无需额外新增 anchor
        #   (注: 一阶差分形式对速度是 O(dt) 精度, 是既有协议, 不改变消融口径)
        pred = jax.vmap(model)(u0)  # (B,1,T,Nx,Ny), 对应 t1..tT
        if pred.ndim != 5 or pred.shape[1] != 1:
            raise ValueError(f"Wave IC-ut expects (B,1,T,Nx,Ny), got {pred.shape}")

        t = pred.shape[2]
        if t < 1:
            raise ValueError("Need at least one future step to compute u_t at t=0.")

        tc = getattr(cfg, "Tc", 1.0)
        dt = 1.0 / t  # normalized tau step

        u1 = pred[:, 0, 0, :, :]      # (B,Nx,Ny), tau=dt
        u_init = u0[:, 0, :, :]       # (B,Nx,Ny), tau=0

        # u_t(physical) = (1/tc) * du/dtau
        ut0 = (u1 - u_init) / (dt * tc)
        loss = jnp.mean(ut0**2)
        return loss, {}

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B,1,Nx,Ny)
        cfg: Any,
        active_losses: Tuple[str, ...] = ("loss_pde", "loss_ic_ut"),
        alpha_w: float = 1.0,
        last_weights: jnp.ndarray | None = None,
        weight_coef: jnp.ndarray | None = None,
        **kwargs,
    ):
        losses = []
        grads = []
        aux_vars: Dict[str, jnp.ndarray] = {}

        for name in active_losses:
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn, has_aux=True)
            (loss, aux), grad = vg_fn(model, u0=u0, cfg=cfg, **kwargs)
            grad = jax.tree.map(lambda g: jnp.nan_to_num(g), grad)
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux)

        n = len(active_losses)
        if getattr(cfg, "use_gradnorm", True):
            weights = self.grad_norm_weights(grads)
        else:
            weights = jnp.ones((n,), dtype=losses[0].dtype)

        if weight_coef is None:
            # 允许从 config 指定固定权重 (use_gradnorm=False 时即最终权重,
            # True 时作为 gradnorm 之上的额外系数)
            weight_coef = getattr(cfg, "weight_coef", None)
        if weight_coef is None:
            weight_coef = jnp.ones((n,), dtype=weights.dtype)
        else:
            weight_coef = jnp.asarray(weight_coef, dtype=weights.dtype)[:n]
        weights = weights * weight_coef

        if last_weights is None:
            last_weights = weights
        else:
            last_weights = jnp.asarray(last_weights, dtype=weights.dtype)[:n]
            weights = alpha_w * weights + (1.0 - alpha_w) * last_weights

        loss_arr = jnp.stack(losses)
        total_loss = jnp.sum(loss_arr * weights)

        total_grad = jax.tree.map(
            lambda *gs: jnp.sum(jnp.stack([w * g for w, g in zip(weights, gs)]), axis=0),
            *grads,
        )

        return (total_loss, (losses, weights, aux_vars)), total_grad

    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            r, _ = ravel_pytree(pytree)
            return jnp.linalg.norm(r)

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1.0 / eps)
        weights = jnp.mean(grad_norms) / grad_norms
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1.0 / eps)
        return jax.lax.stop_gradient(weights)


if __name__ == "__main__":
    import time as _time

    B, C, Nx, Ny, T = 2, 1, 16, 16, 20
    model = FNO(
        key=jax.random.PRNGKey(0),
        in_channels=C,
        out_channels=C,
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

    cfg = type("Config", (), {
        "Lc": 1.0, "Tc": 1.0, "lx": 1.0, "ly": 1.0,
        "active_losses": ("loss_pde", "loss_ic_ut"),
    })

    # 1) fd / ad 两种 time_scheme 的 loss + 梯度冒烟测试（含 causal 路径）
    causal_w = CausalWeightor(num_chunks=5, t_range=(0.0, 1.0))
    for scheme, extra, cw in (
        ("fd", {}, None),
        ("ad", {}, None),
        ("ad", {"use_causality": True, "use_gradnorm": False,
                "weight_coef": (1.0, 100.0)}, causal_w),
    ):
        cfg_i = type("Config", (), {
            **vars(cfg), "use_causality": cw is not None, **extra,
        })
        losses = Losses(causal_weightor=cw, time_scheme=scheme)
        t0 = _time.perf_counter()
        (total, (losses_v, weights_v, aux)), grad = losses.loss_fn(
            model, u0=u0, cfg=cfg_i, active_losses=("loss_pde", "loss_ic_ut"),
            causal_eps=jnp.array(1e-2),
        )
        total.block_until_ready()
        grads_finite = jax.tree_util.tree_all(
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grad)
        )
        print(
            f"[{scheme} {extra or 'gradnorm'}] total={float(total):.4e}  "
            f"pde={float(losses_v[0]):.4e} ic_ut={float(losses_v[1]):.4e}  "
            f"w={[float(w) for w in weights_v]}  "
            f"grad_finite={bool(grads_finite)}  "
            f"({_time.perf_counter() - t0:.1f}s incl. compile)"
        )

    # 2) AD 二阶时间导数正确性: 与 uniform-shift 的二阶中心差分对拍
    #    (float32 下差分舍入会放大, 故临时开 x64 复验)
    jax.config.update("jax_enable_x64", True)
    model64 = jax.tree.map(
        lambda a: a.astype(jnp.float64) if isinstance(a, jax.Array) else a, model
    )
    u0_64 = u0.astype(jnp.float64)
    _, utt64 = _time_second_derivative_ad(model64, u0_64[0], T)
    t_base = jnp.arange(1, T + 1, dtype=jnp.float64) / T
    for delta in (1e-2, 1e-3):
        f_p = model64(u0_64[0], t_coords=t_base + delta)
        f_m = model64(u0_64[0], t_coords=t_base - delta)
        f_0 = model64(u0_64[0], t_coords=t_base)
        fd = (f_p + f_m - 2.0 * f_0) / (delta ** 2)
        rel = jnp.linalg.norm(utt64 - fd) / jnp.linalg.norm(fd)
        print(
            f"AD 2nd deriv vs shift-FD (float64, delta={delta:.0e}): "
            f"relative diff = {float(rel):.3e}"
        )
