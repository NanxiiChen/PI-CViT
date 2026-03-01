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
    x: (B, T, Nx, Ny), T>=3
    二阶差分近似 d2x/dtau2 (tau in [0,1])
    """
    xtt = jnp.zeros_like(x)
    xtt = xtt.at[:, 1:-1].set((x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / (dt**2))
    # 边界使用单边二阶
    xtt = xtt.at[:, 0].set((x[:, 2] - 2.0 * x[:, 1] + x[:, 0]) / (dt**2))
    xtt = xtt.at[:, -1].set((x[:, -1] - 2.0 * x[:, -2] + x[:, -3]) / (dt**2))
    return xtt


class Losses(eqx.Module):
    causal_weightor: CausalWeightor | None = eqx.field(static=True, default=None)

    def __init__(self, causal_weightor: CausalWeightor | None = None, **kwargs):
        self.causal_weightor = causal_weightor

    @eqx.filter_jit
    def loss_pde(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, 1, Nx, Ny), t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
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
        u = pred_full[:, 0, :, :, :]  # (B,T+1,Nx,Ny)

        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)

        dt = 1.0 / t  # normalized tau step

        # c(x,y)
        x = jnp.linspace(0.0, lx, nx, endpoint=False, dtype=u.dtype)
        y = jnp.linspace(0.0, ly, ny, endpoint=False, dtype=u.dtype)
        xx, yy = jnp.meshgrid(x, y, indexing="ij")  # (Nx, Ny)
        c_xy = 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * xx / lx) * jnp.sin(2.0 * jnp.pi * yy / ly)
        c2 = c_xy**2  # (Nx, Ny)

        utt = _time_second_derivative_fd(u, dt=dt)      # d2u/dtau2
        lap = _spectral_lap_2d(u, lx=lx, ly=ly)

        # d2u/dt2 = (1/tc^2) d2u/dtau2
        residual = utt / (tc**2) - c2[None, None, :, :] * lap / (lc**2)  # (B,T+1,Nx,Ny)
        residual = residual / 100.0 # scale for numerical stability

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
        weights = self.grad_norm_weights(grads)

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
