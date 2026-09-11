"""
Compare PDE-residual MSE losses (no causal weighting) on the SAME FEM space-time grid
(100 test cases x 31 snapshots x 64x64 points), all in pure jax/jnp (float32):

  (0) FEM reference snapshots, derivatives by finite differences (baseline, computed once);
  per trained checkpoint:
  (+) model evaluated at ALL FEM grid points, derivatives by autodiff
      (exactly the training-path estimator Losses.residual_pde);
  (+) the same model values at the same grid points, derivatives by the same finite
      differences as (0);
  (+) relative L2 error of the model vs the FEM reference over all snapshots.

Usage:
    python ice_melting/compare_model_fd_residual.py \
        --ckpts PATH_A PATH_B [--tags adam soap] [--case-batch 50]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax
import equinox as eqx
import jax.numpy as jnp

from models import get_model

from ice_melting.configs import load_configs
from ice_melting.losses import Losses
from ice_melting.fem_pde_loss import LC, TC, M_VAL, LAMBDA, EPSILON

DEFAULT_CKPTS = [
    "/root/autodl-tmp/tf-logs/ice_melting/ellipse/cvit/sota-0121/model_epoch_14500.eqx",
    "/root/autodl-tmp/tf-logs/ice_melting/ellipse/cvit/ablation/soap/soap-baseline/model_epoch_14500.eqx",
]
DATA_DIR = "/root/autodl-tmp/data/100testdata"


def d2_last(f: jax.Array, d: float) -> jax.Array:
    """2nd derivative along the last axis: central interior, 2nd-order one-sided ends."""
    d2 = jnp.zeros_like(f)
    d2 = d2.at[..., 1:-1].set((f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / d**2)
    d2 = d2.at[..., 0].set((2.0 * f[..., 0] - 5.0 * f[..., 1] + 4.0 * f[..., 2] - f[..., 3]) / d**2)
    d2 = d2.at[..., -1].set((2.0 * f[..., -1] - 5.0 * f[..., -2] + 4.0 * f[..., -3] - f[..., -4]) / d**2)
    return d2


def d1_time(f: jax.Array, d: float) -> jax.Array:
    """1st derivative along axis 1 (time) of (B, T, H, W): central + one-sided ends."""
    d1 = jnp.zeros_like(f)
    d1 = d1.at[:, 1:-1].set((f[:, 2:] - f[:, :-2]) / (2.0 * d))
    d1 = d1.at[:, 0].set((-3.0 * f[:, 0] + 4.0 * f[:, 1] - f[:, 2]) / (2.0 * d))
    d1 = d1.at[:, -1].set((3.0 * f[:, -1] - 4.0 * f[:, -2] + f[:, -3]) / (2.0 * d))
    return d1


def residual(phi: jax.Array, dphi_dt: jax.Array, lap: jax.Array) -> jax.Array:
    """Physical-unit residual, same form as Losses.residual_pde."""
    F = (phi**2 - 1.0) ** 2 / 4.0
    dF_dphi = phi**3 - phi
    sqrt_2F = jnp.sqrt(2.0 * F)
    return dphi_dt - M_VAL * (lap - dF_dphi / EPSILON**2) + LAMBDA * sqrt_2F / EPSILON


def fd_residual(phi: jax.Array, times: jax.Array, dx: float, dy: float) -> jax.Array:
    """FD residual via the normalized-coordinate path: FD w.r.t. (x/Lc, t/Tc), then /Tc, /Lc^2."""
    dt = times[1] - times[0]
    lap_hat = jnp.swapaxes(d2_last(jnp.swapaxes(phi, -1, -2), dy / LC), -1, -2) + d2_last(phi, dx / LC)
    dphi_dt_hat = d1_time(phi, dt / TC)
    return residual(phi, dphi_dt_hat / TC, lap_hat / LC**2)  # (B, T, H, W)


def evaluate_model(ckpt, tag, cfg, key, us, params, x_grid, times, phi_ref, dx, dy,
                   H, W, case_batch):
    """Autodiff / FD residual MSE and rel. L2 error of one checkpoint on the FEM grid."""
    model = get_model(cfg.model_name, key, **cfg.model_params)
    model = eqx.tree_deserialise_leaves(ckpt, model)
    print(f"\n=== {tag}: loaded {ckpt} ===", flush=True)
    B, T = us.shape[0], times.shape[0]

    # (autodiff) training-path residual at every FEM grid point
    losses = Losses()
    residual_fn = jax.vmap(
        losses.residual_pde, in_axes=(None, 0, 0, None, None, None)
    )  # maps over case batch, inner vmapped over N_query
    res_ad = jnp.zeros((B, T, H * W), dtype=jnp.float32)
    for ti in range(T):
        tq = jnp.full((x_grid.shape[0], 1), times[ti] / TC)  # (Nq, 1)
        for i in range(0, B, case_batch):
            r = residual_fn(model, us[i : i + case_batch], params[i : i + case_batch],
                            x_grid, tq, cfg)  # (b, Nq)
            res_ad = res_ad.at[i : i + case_batch, ti].set(r)
        print(f"  autodiff residuals: t={float(times[ti]):.1f} ({ti + 1}/{T}) done", flush=True)
    mse_ad = jnp.mean(res_ad**2)

    # model values on the grid
    enc = jax.vmap(model.encoder)(us)  # (B, N_patch, emb)

    @jax.jit
    def decode_at_t(enc_out, t):
        tq = jnp.full((x_grid.shape[0], 1), t)  # (Nq, 1)
        return jax.vmap(model.decoder, in_axes=(0, None, None))(enc_out, x_grid, tq)  # (b, Nq, 1)

    # decode in time/case chunks: one giant fused kernel over T*B*Nq crashes ptxas
    phi_model = jnp.zeros((B, T, H, W), dtype=jnp.float32)
    for ti in range(T):
        for i in range(0, B, case_batch):
            sol_t = decode_at_t(enc[i : i + case_batch], times[ti] / TC)
            phi_model = phi_model.at[i : i + case_batch, ti].set(
                sol_t[..., 0].reshape(-1, H, W)
            )

    # (FD) same stencils as the FEM baseline, applied to the model field
    res_model_fd = fd_residual(phi_model, times, dx, dy)  # (B, T, H, W)
    mse_model_fd = jnp.mean(res_model_fd**2)

    # relative L2 error vs FEM over all snapshots
    l2 = jnp.sqrt(jnp.sum((phi_ref - phi_model) ** 2, axis=(1, 2, 3))) / jnp.sqrt(
        jnp.sum(phi_ref**2, axis=(1, 2, 3))
    )
    gy, gx = jnp.gradient(phi_model, dy, dx, axis=(-2, -1))
    sharp_model = jnp.max(jnp.sqrt(gx**2 + gy**2), axis=(-2, -1)).mean()

    return dict(
        mse_ad=float(mse_ad),
        mse_fd=float(mse_model_fd),
        l2_mean=float(l2.mean()),
        l2_max=float(l2.max()),
        sharp=float(sharp_model),
        prof_ad=np.asarray((res_ad**2).mean(axis=(0, 2))),
        prof_fd=np.asarray((res_model_fd**2).mean(axis=(0, 2, 3))),
        l2_per_case=np.asarray(l2),
        phi_model=np.asarray(phi_model),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    parser.add_argument("--tags", nargs="+", default=None,
                        help="display names matching --ckpts (default: checkpoint dir name)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--case-batch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tags = args.tags if args.tags else [Path(c).parent.name for c in args.ckpts]
    assert len(tags) == len(args.ckpts)

    cfg = load_configs("train_cvit")
    key = jax.random.PRNGKey(args.seed)

    # ---- data (cast to float32 jnp; computation stays on device) ----
    sol = jnp.asarray(np.load(f"{args.data_dir}/solutions.npy"), dtype=jnp.float32)  # (B,T,1,H,W)
    times = jnp.asarray(np.load(f"{args.data_dir}/times.npy"), dtype=jnp.float32)  # (T,)
    mesh = jnp.asarray(np.load(f"{args.data_dir}/mesh_points.npy"), dtype=jnp.float32)  # (2,H,W)
    params = jnp.asarray(np.load(f"{args.data_dir}/initial_params.npy"), dtype=jnp.float32)
    B, T, C, H, W = sol.shape
    dx = float(mesh[0, 0, 1] - mesh[0, 0, 0])
    dy = float(mesh[1, 1, 0] - mesh[1, 0, 0])
    phi_ref = sol[:, :, 0]  # (B, T, H, W)
    us = sol[:, 0]  # (B, 1, H, W) IC snapshot as model input
    x_grid = jnp.stack([mesh[0].reshape(-1), mesh[1].reshape(-1)], axis=-1) / LC  # (Nq, 2)
    print(f"Data: {args.data_dir}: {B} cases, {sol.shape}, dx={dx:.4f}, dt={float(times[1]-times[0]):g}")

    # ================= (0) FEM reference + FD =================
    res_fem_fd = fd_residual(phi_ref, times, dx, dy)  # (B, T, H, W)
    mse_fem_fd = float(jnp.mean(res_fem_fd**2))
    prof_fem = np.asarray((res_fem_fd**2).mean(axis=(0, 2, 3)))
    gy, gx = jnp.gradient(phi_ref, dy, dx, axis=(-2, -1))
    sharp_fem = float(jnp.max(jnp.sqrt(gx**2 + gy**2), axis=(-2, -1)).mean())

    # ================= per-model evaluation =================
    results = {}
    for tag, ckpt in zip(tags, args.ckpts):
        results[tag] = evaluate_model(
            ckpt, tag, cfg, key, us, params, x_grid, times, phi_ref, dx, dy,
            H, W, args.case_batch,
        )

    # ================= summary =================
    print("\n==================== Summary (same grid points, no causal) ====================")
    print(f"{'model':<16} {'FD MSE':>11} {'autodiff MSE':>13} {'L2 err (mean)':>14} {'L2 err (max)':>13}")
    print(f"{'FEM reference':<16} {mse_fem_fd:>11.4e} {'-':>13} {'-':>14} {'-':>13}")
    for tag, r in results.items():
        print(f"{tag:<16} {r['mse_fd']:>11.4e} {r['mse_ad']:>13.4e} "
              f"{r['l2_mean']:>14.4e} {r['l2_max']:>13.4e}")
    print("================================================================================")
    print(f"Interface sharpness (mean max|grad(phi)|): FEM={sharp_fem:.4f}, "
          + ", ".join(f"{t}={r['sharp']:.4f}" for t, r in results.items()))

    print(f"\nPer-time residual MSE profiles:")
    print(f"{'t':>5} {'FEM+FD':>11} " + " ".join(f"{t+'+FD':>12} {t+'+AD':>12}" for t in results))
    for ti in range(0, T, 3):
        row = f"{float(times[ti]):>5.1f} {prof_fem[ti]:>11.4e} "
        for r in results.values():
            row += f"{r['prof_fd'][ti]:>12.4e} {r['prof_ad'][ti]:>12.4e} "
        print(row)

    out = Path(args.data_dir) / "model_vs_fem_mse_adam_soap.npz"
    np.savez(
        out,
        mse_fem_fd=mse_fem_fd,
        sharpness_fem=sharp_fem,
        prof_fem=prof_fem,
        times=np.asarray(times),
        tags=np.array(tags),
        **{f"{tag}/{k}": v for tag, r in results.items() for k, v in r.items()},
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
