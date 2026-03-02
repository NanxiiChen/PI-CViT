from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from models.causal import CausalWeightor
from models.fno import FNO


def _fft_k(n: int, length, dtype=jnp.float32):
    # cycles -> radians
    # note in AC equation，we generate the mesh using jnp.linspace(0, L, N, endpoint=True)
    # which is different from the burgers equation where we use endpoint=False.
    # So the effective length for fftfreq is length/(n-1) instead of length/n.
    return (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=length / (n - 1)).astype(dtype)


def _spectral_lap_2d(u: jnp.ndarray, lx, ly):
    """
    u: (..., Nx, Ny)
    return: lap with same shape
    """
    nx, ny = u.shape[-2], u.shape[-1]
    u_hat = jnp.fft.fftn(u, axes=(-2, -1))

    kx = _fft_k(nx, lx, dtype=u.dtype)  # (Nx,)
    ky = _fft_k(ny, ly, dtype=u.dtype)  # (Ny,)
    lap_mul = -(kx[:, None] ** 2 + ky[None, :] ** 2)  # (Nx, Ny)

    lap = jnp.fft.ifftn(u_hat * lap_mul, axes=(-2, -1)).real
    return lap


def _fd2_lap_2d_neumann(u: jnp.ndarray, lx, ly):
    """
    二阶差分拉普拉斯 + Neumann(du/dn=0) 边界
    u: (..., Nx, Ny)
    """
    nx, ny = u.shape[-2], u.shape[-1]
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    # d2/dx2
    u_xx = jnp.zeros_like(u)
    u_xx = u_xx.at[..., 1:-1, :].set((u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]) / (dx**2))
    # Neumann 边界（镜像 ghost 点）
    u_xx = u_xx.at[..., 0, :].set(2.0 * (u[..., 1, :] - u[..., 0, :]) / (dx**2))
    u_xx = u_xx.at[..., -1, :].set(2.0 * (u[..., -2, :] - u[..., -1, :]) / (dx**2))

    # d2/dy2
    u_yy = jnp.zeros_like(u)
    u_yy = u_yy.at[..., :, 1:-1].set((u[..., :, 2:] - 2.0 * u[..., :, 1:-1] + u[..., :, :-2]) / (dy**2))
    # Neumann 边界（镜像 ghost 点）
    u_yy = u_yy.at[..., :, 0].set(2.0 * (u[..., :, 1] - u[..., :, 0]) / (dy**2))
    u_yy = u_yy.at[..., :, -1].set(2.0 * (u[..., :, -2] - u[..., :, -1]) / (dy**2))

    return u_xx + u_yy


def _lap_2d(u: jnp.ndarray, lx, ly, spatial_scheme: str):
    if spatial_scheme == "fft":
        return _spectral_lap_2d(u, lx=lx, ly=ly)
    if spatial_scheme == "fd":
        return _fd2_lap_2d_neumann(u, lx=lx, ly=ly)
    raise ValueError(f"spatial_scheme must be 'fft' or 'fd', got {spatial_scheme}")


def _time_derivative_fd(x: jnp.ndarray, dt):
    """
    x: (B, T, Nx, Ny)
    """
    xt = jnp.zeros_like(x)
    xt = xt.at[:, 1:-1].set((x[:, 2:] - x[:, :-2]) / (2.0 * dt))
    xt = xt.at[:, 0].set((x[:, 1] - x[:, 0]) / dt)
    xt = xt.at[:, -1].set((x[:, -1] - x[:, -2]) / dt)
    return xt


def _allen_cahn_rhs_normalized_time(
    phi: jnp.ndarray,
    M_val,
    lbd,
    epsilon,
    lc,
    tc,
    lx,
    ly,
    spatial_scheme: str = "fft",
):
    """
    phi: (..., Nx, Ny), normalized time tau in [0,1]

    PDE:
      phi_t = M * (Delta phi - (phi^3 - phi)/epsilon^2) - lbd * sqrt(2F(phi))/epsilon
      F(phi) = (phi^2 - 1)^2 / 4

    with t = tc * tau and x = lc * x_hat:
      dphi/dtau = tc * [ M * (Delta phi / lc^2 - (phi^3 - phi)/epsilon^2)
                         - lbd * sqrt(2F(phi))/epsilon ]
    """
    lap = _lap_2d(phi, lx=lx, ly=ly, spatial_scheme=spatial_scheme)
    F_phi = 0.25 * (phi**2 - 1.0) ** 2
    dF_dphi = phi**3 - phi
    source = jnp.sqrt(jnp.clip(2.0 * F_phi, a_min=0.0)) / epsilon

    rhs = tc * (M_val * (lap / (lc**2) - dF_dphi / (epsilon**2)) - lbd * source)
    return rhs


class Losses(eqx.Module):
    causal_weightor: CausalWeightor | None = eqx.field(static=True, default=None)
    time_scheme: str = eqx.field(static=True, default="rk4")   # "rk4" | "fd"
    spatial_scheme: str = eqx.field(static=True, default="fft")  # "fft" | "fd2"

    def __init__(
        self,
        causal_weightor: CausalWeightor | None = None,
        time_scheme: str = "rk4",
        spatial_scheme: str = "fft",
    ):
        if time_scheme not in ("rk4", "fd"):
            raise ValueError("time_scheme must be 'rk4' or 'fd'")
        if spatial_scheme not in ("fft", "fd"):
            raise ValueError("spatial_scheme must be 'fft' or 'fd'")
        self.causal_weightor = causal_weightor
        self.time_scheme = time_scheme
        self.spatial_scheme = spatial_scheme

    @eqx.filter_jit
    def loss_pde(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny), t=0
        cfg: Any,
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        pred = jax.vmap(model)(u0)  # (B,1,T,Nx,Ny), 对应 t1..tT
        if pred.ndim != 5:
            raise ValueError(f"FNO output must be (B,C,T,Nx,Ny), got {pred.shape}")
        if pred.shape[1] != 1:
            raise ValueError(f"Allen-Cahn expects C=1, got C={pred.shape[1]}")

        _, _, t, nx, ny = pred.shape
        if t < 1:
            raise ValueError("pred must contain at least one future step")

        pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,1,T+1,Nx,Ny)
        phi = pred_full[:, 0, :, :, :]  # (B,T+1,Nx,Ny)

        M_val = getattr(cfg, "M_val", 0.1)
        lbd = getattr(cfg, "lbd", getattr(cfg, "lambda_val", 5.0))
        epsilon = getattr(cfg, "epsilon", 1.0)

        lc = getattr(cfg, "Lc", 100.0)
        tc = getattr(cfg, "Tc", 3.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        spatial_scheme = getattr(cfg, "spatial_scheme", getattr(cfg, "space_scheme", self.spatial_scheme))

        dt_phys = getattr(cfg, "dt", None)
        dt = (dt_phys / tc) if (dt_phys is not None) else (1.0 / t)

        if self.time_scheme == "fd":
            phit = _time_derivative_fd(phi, dt=dt)
            lap = _lap_2d(phi, lx=lx, ly=ly, spatial_scheme=spatial_scheme)

            F_phi = 0.25 * (phi**2 - 1.0) ** 2
            dF_dphi = phi**3 - phi
            source = jnp.sqrt(jnp.clip(2.0 * F_phi, a_min=0.0)) / epsilon

            residual = phit / tc - M_val * (lap / (lc**2) - dF_dphi / (epsilon**2)) + lbd * source
            res_sq_t = jnp.mean(residual[:, 1:] ** 2, axis=(-2, -1))  # (B,T)
            ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        else:  # rk4
            s_n = phi[:, :-1, :, :]
            s_np1 = phi[:, 1:, :, :]

            rhs = lambda s: _allen_cahn_rhs_normalized_time(
                s,
                M_val=M_val,
                lbd=lbd,
                epsilon=epsilon,
                lc=lc,
                tc=tc,
                lx=lx,
                ly=ly,
                spatial_scheme=spatial_scheme,
            )

            k1 = rhs(s_n)
            k2 = rhs(s_n + 0.5 * dt * k1)
            k3 = rhs(s_n + 0.5 * dt * k2)
            k4 = rhs(s_n + dt * k3)

            s_rk4 = s_n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            defect = s_np1 - s_rk4

            res_sq_t = jnp.mean(defect**2, axis=(-2, -1))
            ts = (jnp.arange(1, t + 1, dtype=pred.dtype) * dt).reshape(-1, 1)

        if bool(getattr(cfg, "use_causality", False)) and self.causal_weightor is not None:
            eps = kwargs.get("causal_eps", None)
            residuals_for_causal = jnp.sqrt(jnp.clip(res_sq_t, a_min=1e-30))
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            return loss, {
                "loss_chunks": loss_chunks,
                "causal_weights": causal_weights,
                "residual_t_mean": jnp.mean(res_sq_t, axis=0),
            }

        return jnp.mean(res_sq_t), {"residual_t_mean": jnp.mean(res_sq_t, axis=0)}

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,  # (B,1,Nx,Ny)
        cfg: Any,
        **kwargs,
    ):
        vg_fn = eqx.filter_value_and_grad(self.loss_pde, has_aux=True)
        (loss, aux), grad = vg_fn(model, u0=u0, cfg=cfg, **kwargs)
        return (loss, aux), grad