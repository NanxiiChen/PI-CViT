from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from models.causal import CausalWeightor
from models.fno import FNO


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
    time_scheme: str = eqx.field(static=True, default="rk4")  # "rk4" | "fd"

    def __init__(
        self,
        causal_weightor: CausalWeightor | None = None,
        time_scheme: str = "rk4",
    ):
        if time_scheme not in ("rk4", "fd"):
            raise ValueError("time_scheme must be 'rk4' or 'fd'")
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
        pred = jax.vmap(model)(u0)  # (B, C, T, Nx, Ny), 统一约定：对应 t1..tT
        if pred.ndim != 5:
            raise ValueError(f"FNO output must be (B,C,T,Nx,Ny), got {pred.shape}")
        if pred.shape[1] != 2:
            raise ValueError(f"Burgers expects C=2, got C={pred.shape[1]}")

        b, c, t, Nx, Ny = pred.shape
        if t < 1:
            raise ValueError("pred must contain at least one future step")

        # 统一约定：显式拼接 t=0 初值，形成完整时间序列 [t0, t1, ..., tT]
        pred_full = jnp.concatenate([u0[:, :, None, :, :], pred], axis=2)  # (B,C,T+1,Nx,Ny)

        nu = getattr(cfg, "nu", 0.01)
        lc = getattr(cfg, "Lc", 1.0)
        tc = getattr(cfg, "Tc", 1.0)
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)

        # 统一时间步长：pred 有 T 帧未来时间 => dt = 1/T
        dt = 1.0 / t

        if self.time_scheme == "fd":
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
            residuals_for_causal = jnp.sqrt(jnp.clip(res_sq_t, a_min=1e-30))
            loss, loss_chunks, causal_weights = self.causal_weightor.compute_causal_loss(
                residuals_for_causal, ts, eps=eps
            )
            aux = {
                "loss_chunks": loss_chunks,
                "causal_weights": causal_weights,
                "residual_t_mean": jnp.mean(res_sq_t, axis=0),
            }
            return loss, aux

        loss = jnp.mean(res_sq_t)
        return loss, {"residual_t_mean": jnp.mean(res_sq_t, axis=0)}

    def loss_fn(
        self,
        model: FNO,
        u0: jnp.ndarray,   # (B, C, Nx, Ny)
        cfg: Any,
        **kwargs,
    ):
        vg_fn = eqx.filter_value_and_grad(self.loss_pde, has_aux=True)
        (loss, aux), grad = vg_fn(model, u0=u0, cfg=cfg, **kwargs)
        return (loss, aux), grad
    
    
if __name__ == "__main__":
    from models.causal import CausalWeightor
    from models.fno import FNO

    B, C, Nx, Ny = 16, 2, 16, 16
    T = 100
    u0 = jnp.zeros((B, C, Nx, Ny))
    key = jax.random.PRNGKey(0)
    model = FNO(
        key=key,
        in_channels=2,
        out_channels=2,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=32,
        depth=4,
        add_coords=True,
        padding=(10, 0, 0),
    )
    # causal_weightor = CausalWeightor(num_chunks=5)
    losses = Losses(causal_weightor=None, time_scheme="rk4")

    cfg = type("Config", (), {"nu": 0.01, "Lc": 1.0, "Tc": 1.0, "dt": 0.1})
    (loss_value, aux), grad = losses.loss_fn(model, u0=u0, cfg=cfg)
    print("Loss:", loss_value.shape)
    print("Aux:", aux.keys())