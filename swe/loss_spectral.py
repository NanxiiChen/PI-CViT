from typing import Any, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from models.causal import CausalWeightor
from models.fno import FNO


def _fft_k(n: int, length, dtype=jnp.float32):
    return (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=length / n).astype(dtype)


def _spectral_grad_2d(x: jnp.ndarray, lx, ly):
    """
    x: (..., Nx, Ny)
    return: dx, dy with same shape
    """
    nx, ny = x.shape[-2], x.shape[-1]
    x_hat = jnp.fft.fftn(x, axes=(-2, -1))
    kx = _fft_k(nx, lx, dtype=x.dtype)
    ky = _fft_k(ny, ly, dtype=x.dtype)
    dx = jnp.fft.ifftn(x_hat * (1j * kx[:, None]), axes=(-2, -1)).real
    dy = jnp.fft.ifftn(x_hat * (1j * ky[None, :]), axes=(-2, -1)).real
    return dx, dy


def _time_derivative_fd(x: jnp.ndarray, dt):
    """
    x: (B, T, Nx, Ny)
    """
    xt = jnp.zeros_like(x)
    xt = xt.at[:, 1:-1].set((x[:, 2:] - x[:, :-2]) / (2.0 * dt))
    xt = xt.at[:, 0].set((x[:, 1] - x[:, 0]) / dt)
    xt = xt.at[:, -1].set((x[:, -1] - x[:, -2]) / dt)
    return xt


class Losses(eqx.Module):
    causal_weightor: CausalWeightor | None = eqx.field(static=True, default=None)

    def __init__(self, causal_weightor: CausalWeightor | None = None, **kwargs):
        self.causal_weightor = causal_weightor

    @eqx.filter_jit
    def loss_continuity(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,3,Nx,Ny): (h,u,v) at t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,3,T,Nx,Ny), 对应 t1..tT
        if pred.ndim != 5 or pred.shape[1] != 3:
            raise ValueError(f"SWE expects (B,3,T,Nx,Ny), got {pred.shape}")
        t = pred.shape[2]
        if t < 1:
            raise ValueError("Need T>=1 for temporal derivative in spectral loss.")

        # [t0, t1, ..., tT]
        pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,3,T+1,Nx,Ny)
        h = pred_full[:, 0]  # (B,T+1,Nx,Ny)
        u = pred_full[:, 1]
        v = pred_full[:, 2]

        H = getattr(cfg, "H_val", 1.0)
        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        dt = 1.0 / t

        ht = _time_derivative_fd(h, dt)
        ux, _ = _spectral_grad_2d(u, lx=lx, ly=ly)
        _, vy = _spectral_grad_2d(v, lx=lx, ly=ly)

        r_cont = ht / tc + H * (ux + vy) / lc
        res_sq_t = jnp.mean(r_cont[:, 1:] ** 2, axis=(-2, -1))  # 仅 t1..tT
        ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        if bool(getattr(cfg, "use_causality", False)) and self.causal_weightor is not None:
            eps = kwargs.get("causal_eps", None)
            residuals_for_causal = jnp.sqrt(jnp.clip(res_sq_t, a_min=1e-30))
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            return loss, {
                "loss_chunks_continuity": loss_chunks,
                "causal_weights_continuity": causal_weights,
                "residual_t_mean_continuity": jnp.mean(res_sq_t, axis=0),
            }

        return jnp.mean(res_sq_t), {
            "residual_t_mean_continuity": jnp.mean(res_sq_t, axis=0)
        }

    @eqx.filter_jit
    def loss_momentum(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,3,Nx,Ny): (h,u,v) at t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,3,T,Nx,Ny), 对应 t1..tT
        if pred.ndim != 5 or pred.shape[1] != 3:
            raise ValueError(f"SWE expects (B,3,T,Nx,Ny), got {pred.shape}")
        t = pred.shape[2]
        if t < 1:
            raise ValueError("Need T>=1 for temporal derivative in spectral loss.")

        # [t0, t1, ..., tT]
        pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,3,T+1,Nx,Ny)
        h = pred_full[:, 0]  # (B,T+1,Nx,Ny)
        u = pred_full[:, 1]
        v = pred_full[:, 2]

        f = getattr(cfg, "f_val", 10.0)
        g = getattr(cfg, "g_val", 1.0)
        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        dt = 1.0 / t

        ut = _time_derivative_fd(u, dt)
        vt = _time_derivative_fd(v, dt)
        hx, hy = _spectral_grad_2d(h, lx=lx, ly=ly)

        r_u = ut / tc - f * v + g * hx / lc
        r_v = vt / tc + f * u + g * hy / lc

        res_sq_t = jnp.mean(r_u[:, 1:] ** 2 + r_v[:, 1:] ** 2, axis=(-2, -1))  # 仅 t1..tT
        ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        if bool(getattr(cfg, "use_causality", False)) and self.causal_weightor is not None:
            eps = kwargs.get("causal_eps", None)
            residuals_for_causal = jnp.sqrt(jnp.clip(res_sq_t, a_min=1e-30))
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            return loss, {
                "loss_chunks_momentum": loss_chunks,
                "causal_weights_momentum": causal_weights,
                "residual_t_mean_momentum": jnp.mean(res_sq_t, axis=0),
            }

        return jnp.mean(res_sq_t), {
            "residual_t_mean_momentum": jnp.mean(res_sq_t, axis=0)
        }

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,
        cfg: Any,
        active_losses: Tuple[str, ...] = ("loss_continuity", "loss_momentum"),
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
            weight_coef = jnp.ones((n,), dtype=weights.dtype)
        else:
            weight_coef = jnp.asarray(weight_coef, dtype=weights.dtype)[:n]
        weights = weights * weight_coef

        if last_weights is None:
            last_weights = weights
        else:
            last_weights = jnp.asarray(last_weights, dtype=weights.dtype)[:n]
            weights = alpha_w * weights + (1.0 - alpha_w) * last_weights

        total_loss = jnp.sum(jnp.stack(losses) * weights)
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
