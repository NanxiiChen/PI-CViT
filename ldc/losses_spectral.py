from typing import Any, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from models.fno import FNO


def _fft_k(n: int, length, dtype=jnp.float32):
    if n <= 1:
        raise ValueError(f"n must be > 1, got {n}")
    return (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=length / n).astype(dtype)


def _spectral_grad_lap_2d(x: jnp.ndarray, lx, ly):
    """
    x: (..., Nx, Ny), FNO约定: axis=-2是x, axis=-1是y
    return: dx, dy, lap (same shape as x)
    """
    nx, ny = x.shape[-2], x.shape[-1]
    x_hat = jnp.fft.fftn(x, axes=(-2, -1))

    kx = _fft_k(nx, lx, dtype=x.dtype)  # (Nx,)
    ky = _fft_k(ny, ly, dtype=x.dtype)  # (Ny,)

    ikx = 1j * kx[:, None]
    iky = 1j * ky[None, :]
    lap_mul = -(kx[:, None] ** 2 + ky[None, :] ** 2)

    dx = jnp.fft.ifftn(x_hat * ikx, axes=(-2, -1)).real
    dy = jnp.fft.ifftn(x_hat * iky, axes=(-2, -1)).real
    lap = jnp.fft.ifftn(x_hat * lap_mul, axes=(-2, -1)).real
    return dx, dy, lap


def _crop_interior(x: jnp.ndarray, pad: int):
    if pad <= 0:
        return x
    nx, ny = x.shape[-2], x.shape[-1]
    if 2 * pad >= nx or 2 * pad >= ny:
        return x
    return x[..., pad:-pad, pad:-pad]


class Losses(eqx.Module):
    def __init__(self, **kwargs):
        pass

    @eqx.filter_jit
    def loss_momentum(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny), 常数通道(归一化Re)
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny): (u,v,p)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"LDC-FNO expects (B,3,Nx,Ny), got {pred.shape}")

        u = pred[:, 0]
        v = pred[:, 1]
        p = pred[:, 2]

        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        Lc = getattr(cfg, "Lc", 1.0)
        pde_pad = int(getattr(cfg, "pde_interior_pad", 0))

        # 归一化Re -> 物理Re，再得到 nu=1/Re
        re_norm = jnp.mean(u0[:, 0], axis=(-2, -1))  # (B,)
        if hasattr(cfg, "denormalize_re"):
            re_val = jax.vmap(cfg.denormalize_re)(re_norm)  # (B,)
        else:
            re_val = re_norm
        nu = 1.0 / jnp.clip(re_val, a_min=1e-8)  # (B,)

        ux, uy, lap_u = _spectral_grad_lap_2d(u, lx=lx, ly=ly)
        vx, vy, lap_v = _spectral_grad_lap_2d(v, lx=lx, ly=ly)
        px, py, _ = _spectral_grad_lap_2d(p, lx=lx, ly=ly)

        adv_u = u * ux + v * uy
        adv_v = u * vx + v * vy

        r_u = adv_u / Lc + px / Lc - nu[:, None, None] * lap_u / (Lc**2)
        r_v = adv_v / Lc + py / Lc - nu[:, None, None] * lap_v / (Lc**2)

        r_u_i = _crop_interior(r_u, pde_pad)
        r_v_i = _crop_interior(r_v, pde_pad)
        loss = jnp.mean(r_u_i**2 + r_v_i**2)

        return loss, {
            "re_vals": re_val,
            "residual_momentum_mean": jnp.mean(r_u_i**2 + r_v_i**2),
        }

    @eqx.filter_jit
    def loss_continuity(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny)
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"LDC-FNO expects (B,3,Nx,Ny), got {pred.shape}")

        u = pred[:, 0]
        v = pred[:, 1]
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        pde_pad = int(getattr(cfg, "pde_interior_pad", 0))

        ux, _, _ = _spectral_grad_lap_2d(u, lx=lx, ly=ly)
        _, vy, _ = _spectral_grad_lap_2d(v, lx=lx, ly=ly)

        r_c = ux + vy
        r_c_i = _crop_interior(r_c, pde_pad)
        loss = jnp.mean(r_c_i**2)

        return loss, {
            "residual_continuity_mean": jnp.mean(r_c_i**2),
        }

    @eqx.filter_jit
    def loss_bc_walls(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny)
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        三面壁(左/右/下) no-slip: u=v=0
        """
        pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"LDC-FNO expects (B,3,Nx,Ny), got {pred.shape}")

        u = pred[:, 0]
        v = pred[:, 1]

        # y=0 (bottom)
        ub, vb = u[:, :, 0], v[:, :, 0]
        # x=0 (left), x=lx (right)
        ul, vl = u[:, 0, :], v[:, 0, :]
        ur, vr = u[:, -1, :], v[:, -1, :]

        loss = (
            jnp.mean(ub**2 + vb**2)
            + jnp.mean(ul**2 + vl**2)
            + jnp.mean(ur**2 + vr**2)
        ) / 3.0
        return loss, {}

    @eqx.filter_jit
    def loss_bc_lid(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny)
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        顶盖(y=ly): v=0, u=u_lid(x)
        默认沿用你原loss中的cosh profile；若要常数顶盖速度可自行改成1.0
        """
        pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"LDC-FNO expects (B,3,Nx,Ny), got {pred.shape}")

        u = pred[:, 0]  # (B,Nx,Ny)
        v = pred[:, 1]  # (B,Nx,Ny)

        nx = u.shape[-2]
        lx = getattr(cfg, "lx", 1.0)

        x = jnp.linspace(0.0, lx, nx, endpoint=True, dtype=u.dtype)  # (Nx,)
        x_hat = x / jnp.maximum(lx, 1e-12)
        u_lid_ref = 1.0 - jnp.cosh(50.0 * (x_hat - 0.5)) / jnp.cosh(25.0)  # (Nx,)

        u_top = u[:, :, -1]  # y=ly
        v_top = v[:, :, -1]

        loss = jnp.mean((u_top - u_lid_ref[None, :]) ** 2) + jnp.mean(v_top**2)
        return loss, {}

    @eqx.filter_jit
    def loss_bc_pressure(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny)
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny)
        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(f"LDC-FNO expects (B,3,Nx,Ny), got {pred.shape}")
        p = pred[:, 2]
        loss = jnp.mean(p[:, 0, 0] ** 2)
        return loss, {}

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,
        cfg: Any,
        active_losses: Tuple[str, ...] = (
            "loss_momentum",
            "loss_continuity",
            "loss_bc_walls",
            "loss_bc_lid",
            "loss_bc_pressure",
        ),
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