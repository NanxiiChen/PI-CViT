"""
Compare PDE-residual MSE losses (no causal) using the FINE-dt FEM reference
(every solver step saved, dt = 0.001), evaluated on the coarse time grid.

Time derivatives use central differences over +/- one FINE step (dt = 0.001,
2nd-order one-sided at t=0 and t=T), even though losses are only EVALUATED at
the downsampled times (every --eval-every = 0.1). Spatial derivatives unchanged
(central + one-sided, dx = dy = 100/63).

On the same 8 FEM cases x 31 eval times x 64x64 grid points:

  (0) FEM reference + FD (fine-dt time stencils)  [baseline, computed once]
  per checkpoint:
  (+) model + autodiff at the eval grid points (training-path estimator)
  (+) model decoded at the fine stencil times + FD (same estimator as (0))
  (+) relative L2 error of the model vs FEM at the eval times

Usage:
    python ice_melting/compare_model_fd_residual_finedt.py \
        --ckpts PATH_A PATH_B [--tags adam soap] [--eval-every 0.1]
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
DATA_DIR = "/root/autodl-tmp/data/10testdata_finedt"


def d2_last(f: jax.Array, d: float) -> jax.Array:
    """2nd derivative along the last axis: central interior, 2nd-order one-sided ends."""
    d2 = jnp.zeros_like(f)
    d2 = d2.at[..., 1:-1].set((f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / d**2)
    d2 = d2.at[..., 0].set((2.0 * f[..., 0] - 5.0 * f[..., 1] + 4.0 * f[..., 2] - f[..., 3]) / d**2)
    d2 = d2.at[..., -1].set((2.0 * f[..., -1] - 5.0 * f[..., -2] + 4.0 * f[..., -3] - f[..., -4]) / d**2)
    return d2


def residual(phi: jax.Array, dphi_dt: jax.Array, lap: jax.Array) -> jax.Array:
    """Physical-unit residual, same form as Losses.residual_pde."""
    F = (phi**2 - 1.0) ** 2 / 4.0
    dF_dphi = phi**3 - phi
    sqrt_2F = jnp.sqrt(2.0 * F)
    return dphi_dt - M_VAL * (lap - dF_dphi / EPSILON**2) + LAMBDA * sqrt_2F / EPSILON


def lap_hat_phys(phi: jax.Array, dx: float, dy: float) -> jax.Array:
    """Laplacian w.r.t. normalized coords, scaled back by 1/Lc^2 (residual_pde path)."""
    return (jnp.swapaxes(d2_last(jnp.swapaxes(phi, -1, -2), dy / LC), -1, -2)
            + d2_last(phi, dx / LC)) / LC**2


def time_stencil(get, k, T, dt, pos=None):
    """d/dt at fine index k via central diff over +/-1 fine step; one-sided at the ends.

    `get(i)` returns the field at fine time index i.
    """
    if k == 0:
        return (-3.0 * get(0) + 4.0 * get(1) - get(2)) / (2.0 * dt)
    if k == T - 1:
        return (3.0 * get(T - 1) - 4.0 * get(T - 2) + get(T - 3)) / (2.0 * dt)
    return (get(k + 1) - get(k - 1)) / (2.0 * dt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    parser.add_argument("--tags", nargs="+", default=None,
                        help="display names matching --ckpts (default: checkpoint dir name)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--eval-every", type=float, default=0.1,
                        help="evaluate losses every this much physical time (downsampled)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tags = args.tags if args.tags else [Path(c).parent.name for c in args.ckpts]
    assert len(tags) == len(args.ckpts)

    cfg = load_configs("train_cvit")
    key = jax.random.PRNGKey(args.seed)

    # ---- data ----
    times_np = np.load(f"{args.data_dir}/times.npy")  # (T,) uniform fine grid
    mesh_np = np.load(f"{args.data_dir}/mesh_points.npy")
    params_np = np.load(f"{args.data_dir}/initial_params.npy")
    sol_mmap = np.load(f"{args.data_dir}/solutions.npy", mmap_mode="r")  # (B,T,1,H,W)
    B, T, C, H, W = sol_mmap.shape
    dt_fine = float(times_np[1] - times_np[0])
    dx = float(mesh_np[0, 0, 1] - mesh_np[0, 0, 0])
    dy = float(mesh_np[1, 1, 0] - mesh_np[1, 0, 0])

    step = int(round(args.eval_every / dt_fine))
    eval_idx = np.arange(0, T, step)
    assert eval_idx[-1] == T - 1, "eval grid should end at the final time"
    n_eval = len(eval_idx)
    # fine indices required by the time stencils at the eval times
    need = {0, 1, 2, T - 3, T - 2, T - 1}
    for k in eval_idx:
        need.update({k - 1, k, k + 1})
    need = np.array(sorted(i for i in need if 0 <= i < T))
    pos = {int(i): p for p, i in enumerate(need)}

    times = jnp.asarray(times_np, dtype=jnp.float32)
    us = jnp.asarray(np.asarray(sol_mmap[:, 0]), dtype=jnp.float32)  # (B,1,H,W) IC as input
    params = jnp.asarray(params_np, dtype=jnp.float32)
    x_grid = jnp.asarray(
        np.stack([mesh_np[0].reshape(-1), mesh_np[1].reshape(-1)], axis=-1) / LC,
        dtype=jnp.float32,
    )
    print(f"Data: {args.data_dir}: {sol_mmap.shape}, dt_fine={dt_fine}, "
          f"eval every {step} steps ({n_eval} times), dx={dx:.4f}")

    # FEM fields at the needed fine indices, (B, N_need, H, W)
    phi_need_fem = jnp.asarray(np.asarray(sol_mmap[:, need, 0]), dtype=jnp.float32)

    def fem_at(i):
        return phi_need_fem[:, pos[i]]

    # ================= (0) FEM + FD (fine-dt stencils) =================
    phi_eval_fem = phi_need_fem[:, [pos[int(k)] for k in eval_idx]]  # (B, n_eval, H, W)
    dphi_dt_fem = jnp.stack(
        [time_stencil(fem_at, int(k), T, dt_fine / TC) for k in eval_idx], axis=1
    )  # (B, n_eval, H, W), w.r.t. t_hat; scaled by 1/Tc inside residual
    res_fem_fd = residual(phi_eval_fem, dphi_dt_fem / TC, lap_hat_phys(phi_eval_fem, dx, dy))
    mse_fem_fd = float(jnp.mean(res_fem_fd**2))
    prof_fem = np.asarray((res_fem_fd**2).mean(axis=(0, 2, 3)))
    gy, gx = jnp.gradient(phi_eval_fem, dy, dx, axis=(-2, -1))
    sharp_fem = float(jnp.max(jnp.sqrt(gx**2 + gy**2), axis=(-2, -1)).mean())
    print(f"\n(0) FEM reference + FD (dt={dt_fine} time stencils): MSE = {mse_fem_fd:.6e}")

    # ================= per-model evaluation =================
    losses = Losses()
    residual_fn = jax.vmap(
        losses.residual_pde, in_axes=(None, 0, 0, None, None, None)
    )

    results = {}
    for tag, ckpt in zip(tags, args.ckpts):
        model = get_model(cfg.model_name, key, **cfg.model_params)
        model = eqx.tree_deserialise_leaves(ckpt, model)
        print(f"\n=== {tag}: loaded {ckpt} ===", flush=True)

        # model values at the needed fine times
        enc = jax.vmap(model.encoder)(us)  # (B, N_patch, emb)

        @jax.jit
        def decode_at_t(enc_out, t):
            tq = jnp.full((x_grid.shape[0], 1), t)  # (Nq, 1)
            return jax.vmap(model.decoder, in_axes=(0, None, None))(enc_out, x_grid, tq)

        model_cache = {
            int(i): decode_at_t(enc, jnp.asarray(float(times_np[i]) / TC, dtype=jnp.float32))[
                ..., 0
            ].reshape(B, H, W)
            for i in need
        }

        def model_at(i):
            return model_cache[i]

        phi_eval_model = jnp.stack([model_at(int(k)) for k in eval_idx], axis=1)  # (B,n_eval,H,W)

        # (FD) same fine-dt stencils as the FEM baseline, applied to the model field
        dphi_dt_model = jnp.stack(
            [time_stencil(model_at, int(k), T, dt_fine / TC) for k in eval_idx], axis=1
        )
        res_model_fd = residual(phi_eval_model, dphi_dt_model / TC,
                                lap_hat_phys(phi_eval_model, dx, dy))
        mse_model_fd = float(jnp.mean(res_model_fd**2))

        # (autodiff) training-path residual at the eval grid points
        res_ad = jnp.zeros((B, n_eval, H * W), dtype=jnp.float32)
        for j, k in enumerate(eval_idx):
            tq = jnp.full((x_grid.shape[0], 1), times_np[k] / TC)
            res_ad = res_ad.at[:, j].set(residual_fn(model, us, params, x_grid, tq, cfg))
            print(f"  autodiff residuals: t={times_np[k]:.1f} ({j + 1}/{n_eval}) done", flush=True)
        mse_ad = float(jnp.mean(res_ad**2))

        # rel. L2 error vs FEM at the eval times
        l2 = jnp.sqrt(jnp.sum((phi_eval_fem - phi_eval_model) ** 2, axis=(1, 2, 3))) / jnp.sqrt(
            jnp.sum(phi_eval_fem**2, axis=(1, 2, 3))
        )
        gy, gx = jnp.gradient(phi_eval_model, dy, dx, axis=(-2, -1))
        sharp_model = float(jnp.max(jnp.sqrt(gx**2 + gy**2), axis=(-2, -1)).mean())

        results[tag] = dict(
            mse_fd=mse_model_fd,
            mse_ad=mse_ad,
            l2_mean=float(l2.mean()),
            l2_max=float(l2.max()),
            sharp=sharp_model,
            prof_fd=np.asarray((res_model_fd**2).mean(axis=(0, 2, 3))),
            prof_ad=np.asarray((res_ad**2).mean(axis=(0, 2))),
            l2_per_case=np.asarray(l2),
        )

    # ================= summary =================
    print("\n=============== Summary (fine-dt stencils, eval every 0.1, no causal) ===============")
    print(f"{'model':<16} {'FD MSE':>11} {'autodiff MSE':>13} {'L2 err (mean)':>14} {'L2 err (max)':>13}")
    print(f"{'FEM reference':<16} {mse_fem_fd:>11.4e} {'-':>13} {'-':>14} {'-':>13}")
    for tag, r in results.items():
        print(f"{tag:<16} {r['mse_fd']:>11.4e} {r['mse_ad']:>13.4e} "
              f"{r['l2_mean']:>14.4e} {r['l2_max']:>13.4e}")
    print("=====================================================================================")
    print(f"Interface sharpness (mean max|grad(phi)|): FEM={sharp_fem:.4f}, "
          + ", ".join(f"{t}={r['sharp']:.4f}" for t, r in results.items()))

    print(f"\nPer-time residual MSE profiles:")
    print(f"{'t':>5} {'FEM+FD':>11} " + " ".join(f"{t+'+FD':>12} {t+'+AD':>12}" for t in results))
    for j in range(0, n_eval, 3):
        row = f"{times_np[eval_idx[j]]:>5.1f} {prof_fem[j]:>11.4e} "
        for r in results.values():
            row += f"{r['prof_fd'][j]:>12.4e} {r['prof_ad'][j]:>12.4e} "
        print(row)

    out = Path(args.data_dir) / "model_vs_fem_mse_finedt.npz"
    np.savez(
        out,
        mse_fem_fd=mse_fem_fd,
        sharpness_fem=sharp_fem,
        prof_fem=prof_fem,
        eval_times=times_np[eval_idx],
        tags=np.array(tags),
        **{f"{tag}/{k}": v for tag, r in results.items() for k, v in r.items()},
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
