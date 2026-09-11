"""
Evaluate the PDE residual loss of the FEM reference solutions via finite differences,
following exactly the non-dimensionalization used in training (Losses.residual_pde).

Convention (see configs/train_cvit.py and sample.py):
    x_hat = x_phys / Lc  in [-0.5, 0.5]   (Lc = 100),   t_hat = t_phys / Tc in [0, 1] (Tc = 3)
The model in training is differentiated w.r.t. the *normalized* coordinates, then

    dphi/dt |_phys = (dphi/dt_hat) / Tc,      Laplacian|_phys = (Laplacian_hat) / Lc**2

so that the residual is assembled in physical units (epsilon, M, lambda are physical):

    r = d(phi)/dt - M * ( Laplacian(phi) - dF/dphi / eps**2 ) + lambda * sqrt(2 F) / eps,
    F = (phi**2 - 1)**2 / 4,  dF/dphi = phi**3 - phi

Here the FEM snapshots are differentiated by 2nd-order central finite differences in
the interior and 2nd-order one-sided differences at the domain boundaries (same
stencils as Losses.loss_pde_fd), first w.r.t. normalized coordinates and then scaled
by 1/Tc and 1/Lc**2 exactly as in losses.py. A direct physical-unit FD computation is
kept as a cross-check (must agree to machine precision).

The causal aggregation reuses the training code path models/causal.CausalWeightor
(num_chunks bins over normalized time t_hat in [0, 1], eps = 100).

Usage:
    python ice_melting/fem_pde_loss.py [--data-dir DIR] [--causal-eps 100] [--num-chunks 24]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Physical / discretization constants, must match ice_melting/fem.py and configs/train_cvit.py
LAMBDA = 5.0
M_VAL = 0.1
N_FEM = 63
H_FEM = 100.0 / N_FEM
EPSILON = 6.0 * H_FEM / (2.0 * np.sqrt(2.0) * np.arctanh(0.9))
LC = 100.0  # xc = x / Lc: maps physical [-50, 50] to normalized [-0.5, 0.5]
TC = 3.0  # tc = t / Tc: maps physical [0, 3] to normalized [0, 1]


def d2_fd(f: np.ndarray, d: float, axis: int) -> np.ndarray:
    """2nd derivative along `axis`: central interior, 2nd-order one-sided at both ends."""
    f = np.moveaxis(f, axis, -1)
    d2 = np.empty_like(f)
    d2[..., 1:-1] = (f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / d**2
    d2[..., 0] = (2.0 * f[..., 0] - 5.0 * f[..., 1] + 4.0 * f[..., 2] - f[..., 3]) / d**2
    d2[..., -1] = (2.0 * f[..., -1] - 5.0 * f[..., -2] + 4.0 * f[..., -3] - f[..., -4]) / d**2
    return np.moveaxis(d2, -1, axis)


def d1_fd(f: np.ndarray, d: float, axis: int) -> np.ndarray:
    """1st derivative along `axis`: central interior, 2nd-order one-sided at both ends."""
    f = np.moveaxis(f, axis, -1)
    d1 = np.empty_like(f)
    d1[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * d)
    d1[..., 0] = (-3.0 * f[..., 0] + 4.0 * f[..., 1] - f[..., 2]) / (2.0 * d)
    d1[..., -1] = (3.0 * f[..., -1] - 4.0 * f[..., -2] + f[..., -3]) / (2.0 * d)
    return np.moveaxis(d1, -1, axis)


def residual(phi: np.ndarray, dphi_dt: np.ndarray, lap: np.ndarray) -> np.ndarray:
    """Assemble the physical-unit residual, same form as Losses.residual_pde."""
    F = (phi**2 - 1.0) ** 2 / 4.0
    dF_dphi = phi**3 - phi
    sqrt_2F = np.sqrt(2.0 * F)  # = |phi^2 - 1| / sqrt(2)
    return dphi_dt - M_VAL * (lap - dF_dphi / EPSILON**2) + LAMBDA * sqrt_2F / EPSILON


def fd_causal_losses(
    phi: np.ndarray,
    times: np.ndarray,
    dx: float,
    dy: float,
    num_chunks: int = 24,
    causal_eps: float = 100.0,
    include_t_end: bool = False,
) -> dict:
    """FD PDE residual + causal aggregation (training code path) for fields on the FEM grid.

    Args:
        phi: (B, T, H, W) field values on the uniform physical FEM grid.
        times: (T,) physical snapshot times (uniform spacing).
        dx, dy: physical grid spacings.
    Returns dict with keys: res (B,T,H,W), mse_all, causal_loss, loss_chunks,
    causal_weights, counts.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from models.causal import CausalWeightor

    B, T, H, W = phi.shape
    dt = float(times[1] - times[0])
    dx_hat, dy_hat, dt_hat = dx / LC, dy / LC, dt / TC
    lap_hat = d2_fd(phi, dy_hat, axis=-2) + d2_fd(phi, dx_hat, axis=-1)
    dphi_dt_hat = d1_fd(phi, dt_hat, axis=1)
    res = residual(phi, dphi_dt_hat / TC, lap_hat / LC**2)  # (B, T, H, W)

    ts_q = np.repeat(times / TC, H * W)  # (T*H*W,), t-major to match reshape below
    residuals = res.reshape(B, T * H * W)  # (B, N_query)
    if include_t_end:
        ts_q = np.minimum(ts_q, 1.0 - 1e-12)
    else:
        keep = ts_q < 1.0
        residuals, ts_q = residuals[:, keep], ts_q[keep]

    weightor = CausalWeightor(num_chunks=num_chunks, t_range=(0.0, 1.0))
    causal_loss, loss_chunks, causal_weights = weightor.compute_causal_loss(
        jnp.asarray(residuals), jnp.asarray(ts_q), eps=causal_eps
    )

    bins = np.linspace(0.0, 1.0, num_chunks + 1)
    idx = np.digitize(ts_q, bins) - 1
    return dict(
        res=res,
        mse_all=float(np.mean(res**2)),
        causal_loss=float(causal_loss),
        loss_chunks=np.asarray(loss_chunks),
        causal_weights=np.asarray(causal_weights),
        counts=np.bincount(idx, minlength=num_chunks),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="/root/autodl-tmp/data/100testdata")
    parser.add_argument("--causal-eps", type=float, default=100.0)
    parser.add_argument("--num-chunks", type=int, default=24)
    parser.add_argument(
        "--include-t-end",
        action="store_true",
        help="count the t_hat=1.0 snapshot (clipped just below the last bin edge); "
        "by default it is excluded, mirroring jnp.digitize behavior in training "
        "where t=1 is never sampled exactly.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    sol = np.load(data_dir / "solutions.npy")  # (B, T, 1, H, W)
    times = np.load(data_dir / "times.npy")  # (T,)
    mesh = np.load(data_dir / "mesh_points.npy")  # (2, H, W): [0]=x, [1]=y
    B, T, C, H, W = sol.shape
    assert C == 1
    phi = sol[:, :, 0].astype(np.float64)  # (B, T, H, W)

    # mesh_points: x varies along axis 1 (columns), y along axis 0 (rows)
    dx = float(mesh[0, 0, 1] - mesh[0, 0, 0])
    dy = float(mesh[1, 1, 0] - mesh[1, 0, 0])
    dt = float(times[1] - times[0])
    # non-dimensionalization: x_hat = x / Lc, t_hat = t / Tc
    dx_hat, dy_hat, dt_hat = dx / LC, dy / LC, dt / TC

    print(f"Data: {data_dir}")
    print(f"  solutions {sol.shape}, physical times {times[0]:g}..{times[-1]:g} (dt={dt:g})")
    print(f"  physical grid: dx={dx:.6f}, dy={dy:.6f}")
    print(f"  normalized grid (x/Lc, t/Tc): dx_hat={dx_hat:.6e}, dy_hat={dy_hat:.6e}, dt_hat={dt_hat:.6e}")
    print(f"  M={M_VAL}, lambda={LAMBDA}, epsilon={EPSILON:.6f} (physical), Lc={LC}, Tc={TC}")
    print(f"  causal: num_chunks={args.num_chunks}, eps={args.causal_eps:g}")

    # ---- primary path: FD in normalized coordinates, then scale back as in losses.py ----
    # (dphi/dt)_phys = (dphi/dt_hat) / Tc ;  Lap_phys = Lap_hat / Lc^2
    lap_hat = d2_fd(phi, dy_hat, axis=-2) + d2_fd(phi, dx_hat, axis=-1)
    dphi_dt_hat = d1_fd(phi, dt_hat, axis=1)
    res = residual(phi, dphi_dt_hat / TC, lap_hat / LC**2)

    # ---- cross-check: FD directly in physical units (algebraically identical) ----
    lap_phys = d2_fd(phi, dy, axis=-2) + d2_fd(phi, dx, axis=-1)
    dphi_dt_phys = d1_fd(phi, dt, axis=1)
    res_phys = residual(phi, dphi_dt_phys, lap_phys)
    scale = max(np.abs(res_phys).max(), np.abs(res).max())
    diff = np.abs(res - res_phys).max() / scale
    print(f"\nCross-check normalized-vs-physical FD residual, max rel diff: {diff:.3e}")
    assert diff < 1e-10, "non-dimensionalized and physical residuals disagree!"

    # ---- unweighted reference stats over the full space-time grid ----
    mse_all = float(np.mean(res**2))
    mse_time_interior = float(np.mean(res[:, 1:-1] ** 2))
    mse_space_interior = float(np.mean(res[:, :, 1:-1, 1:-1] ** 2))
    print("\nUnweighted residual MSE (full grid, all 31 snapshots):", f"{mse_all:.6e}")
    print("  time-interior slices only (central dt stencils):    ", f"{mse_time_interior:.6e}")
    print("  space-interior points only (central dx/dy stencils):", f"{mse_space_interior:.6e}")

    # ---- causal loss via the training code path (ts in normalized time) ----
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from models.causal import CausalWeightor

    t_hat = times / TC  # (T,), in [0, 1]
    ts_q = np.repeat(t_hat, H * W)  # (T*H*W,), t-major to match reshape below
    residuals = res.reshape(B, T * H * W)  # (B, N_query)

    if args.include_t_end:
        ts_q = np.minimum(ts_q, 1.0 - 1e-12)
    else:
        keep = ts_q < 1.0
        residuals = residuals[:, keep]
        ts_q = ts_q[keep]

    weightor = CausalWeightor(num_chunks=args.num_chunks, t_range=(0.0, 1.0))
    causal_loss, loss_chunks, causal_weights = weightor.compute_causal_loss(
        jnp.asarray(residuals), jnp.asarray(ts_q), eps=args.causal_eps
    )
    loss_chunks = np.asarray(loss_chunks)
    causal_weights = np.asarray(causal_weights)
    causal_loss = float(causal_loss)

    # counts per chunk and per-sample causal loss (same weights, numpy re-derivation)
    bins = np.linspace(0.0, 1.0, args.num_chunks + 1)
    idx = np.digitize(ts_q, bins) - 1  # (N_query,)
    counts = np.bincount(idx, minlength=args.num_chunks)
    sq = residuals**2
    chunk_sums = np.stack([np.bincount(idx, weights=sq[b], minlength=args.num_chunks) for b in range(B)])
    chunks_per_sample = chunk_sums / (counts[None, :] + 1e-12)  # (B, num_chunks)
    assert np.allclose(chunks_per_sample.mean(axis=0), loss_chunks, rtol=1e-8)
    per_sample_causal = (chunks_per_sample * causal_weights[None, :]).mean(axis=1)

    edges = bins * TC  # chunk edges back in physical time
    print(f"\nCausal PDE loss (eps={args.causal_eps:g}, {args.num_chunks} chunks): {causal_loss:.6e}")
    print(f"Per-sample causal loss over {B} cases: "
          f"mean={per_sample_causal.mean():.6e}, std={per_sample_causal.std():.6e}, "
          f"min={per_sample_causal.min():.6e}, max={per_sample_causal.max():.6e}")
    print(f"\n{'chunk':>5} {'t range':>15} {'#pts':>8} {'chunk loss':>12} {'weight':>10}")
    for i in range(args.num_chunks):
        print(f"{i:>5} [{edges[i]:>5.2f},{edges[i + 1]:>5.2f}] {counts[i] * B:>8} "
              f"{loss_chunks[i]:>12.4e} {causal_weights[i]:>10.4f}")

    # ---- save results + figure ----
    out_npz = data_dir / f"fd_pde_loss_causal_eps{args.causal_eps:g}.npz"
    np.savez(
        out_npz,
        mse_all=mse_all,
        mse_time_interior=mse_time_interior,
        mse_space_interior=mse_space_interior,
        causal_loss=causal_loss,
        causal_eps=args.causal_eps,
        loss_chunks=loss_chunks,
        causal_weights=causal_weights,
        chunk_counts=counts,
        chunk_edges=edges,
        per_sample_causal=per_sample_causal,
    )
    print(f"\nSaved results to {out_npz}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (bins[:-1] + bins[1:]) * TC
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(centers, loss_chunks, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("physical t")
    axes[0].set_ylabel("chunk loss (mean sq. residual)")
    axes[0].set_title("FEM reference: FD PDE loss chunks")
    axes[1].plot(centers, causal_weights, marker="o")
    axes[1].set_xlabel("physical t")
    axes[1].set_ylabel("weight")
    axes[1].set_title(f"causal weights (eps={args.causal_eps:g})")
    fig.tight_layout()
    out_png = data_dir / f"fd_pde_loss_causal_eps{args.causal_eps:g}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
