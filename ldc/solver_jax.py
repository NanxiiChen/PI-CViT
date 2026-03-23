"""
Lid-Driven Cavity Flow Solver (2D, Incompressible) — Pure JAX / GPU Implementation
====================================================================================

Numerical methods:
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport  (implicit diffusion + explicit upwind convection)
- Red–Black SOR for streamfunction Poisson equation

All hot-path computations are:
  • expressed as pure JAX array operations (no Python loops in the inner solver)
  • JIT-compiled with @jax.jit
  • fully compatible with GPU/TPU execution via JAX's XLA backend

Author:  Kartikey Singh (original NumPy version)
JAX port: 2026
License: MIT

Usage
-----
    python lid_driven_cavity_jax.py            # default Re=100, N=101
    python lid_driven_cavity_jax.py --Re 400 --N 129 --maxiter 80000

Requirements
------------
    pip install jax[cuda12]   # for GPU  (or jax[cpu] for CPU-only)
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Optional: force 64-bit floats (better accuracy, ~2× slower on some GPUs)
# Comment out to use float32 (faster, enough for Re < 1000 on coarse grids)
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)

print(f"JAX version : {jax.__version__}")
print(f"Backend     : {jax.default_backend()}")
print(f"Devices     : {jax.devices()}")


# ===========================================================================
# Data container (immutable, lives on device)
# ===========================================================================

class CavityState(NamedTuple):
    """All mutable fields that evolve during the solve loop."""
    psi   : jnp.ndarray   # (N, N)  streamfunction
    omega : jnp.ndarray   # (N, N)  vorticity
    u     : jnp.ndarray   # (N, N)  horizontal velocity
    v     : jnp.ndarray   # (N, N)  vertical velocity


# ===========================================================================
# Thomas Algorithm (TDMA) — vectorised over M independent systems
# ===========================================================================

def tdma_batch(a: jnp.ndarray,
               b: jnp.ndarray,
               c: jnp.ndarray,
               d: jnp.ndarray) -> jnp.ndarray:
    """
    Solve M independent tridiagonal systems simultaneously via lax.scan.

    Parameters
    ----------
    a, b, c : (M, K)  lower / main / upper diagonals  (a[*, 0] is unused)
    d       : (M, K)  right-hand sides

    Returns
    -------
    x : (M, K)  solutions
    """
    M, K = d.shape

    # ---- Forward sweep ----
    def fwd_step(carry, k):
        p_prev, q_prev = carry                    # (M,)
        denom   = b[:, k] - a[:, k] * p_prev
        p_k     = c[:, k] / denom
        q_k     = (d[:, k] - a[:, k] * q_prev) / denom
        return (p_k, q_k), (p_k, q_k)

    p0 = c[:, 0] / b[:, 0]
    q0 = d[:, 0] / b[:, 0]
    _, (P, Q) = jax.lax.scan(fwd_step, (p0, q0), jnp.arange(1, K))
    # P, Q shape: (K-1, M) — prepend the k=0 values
    P = jnp.concatenate([p0[None, :], P], axis=0).T  # (M, K)
    Q = jnp.concatenate([q0[None, :], Q], axis=0).T  # (M, K)

    # ---- Backward substitution ----
    def bwd_step(x_next, k):
        x_k = Q[:, k] - P[:, k] * x_next
        return x_k, x_k

    x_last = Q[:, -1]
    _, X_rev = jax.lax.scan(bwd_step, x_last, jnp.arange(K - 2, -1, -1))
    # X_rev: (K-1, M)
    X = jnp.concatenate([X_rev[::-1], x_last[None, :]], axis=0).T  # (M, K)
    return X


# ===========================================================================
# Boundary conditions
# ===========================================================================

@partial(jax.jit, static_argnums=(2, 3))
def apply_bcs(psi: jnp.ndarray,
              omega: jnp.ndarray,
              h: float,
              U: float) -> jnp.ndarray:
    """
    Apply Thom's formula vorticity BCs; return updated omega.
    psi boundaries are already zero and stay zero — no update needed.
    """
    # Top wall (moving lid)
    omega_top    = -2.0 * psi[-2, 1:-1] / h**2 - 2.0 * U / h
    # Bottom wall
    omega_bot    = -2.0 * psi[ 1, 1:-1] / h**2
    # Left wall
    omega_left   = -2.0 * psi[1:-1,  1] / h**2
    # Right wall
    omega_right  = -2.0 * psi[1:-1, -2] / h**2

    omega = omega.at[-1, 1:-1].set(omega_top)
    omega = omega.at[ 0, 1:-1].set(omega_bot)
    omega = omega.at[1:-1,  0].set(omega_left)
    omega = omega.at[1:-1, -1].set(omega_right)

    # Corners (average of two adjacent wall nodes)
    omega = omega.at[ 0,  0].set(0.5 * (omega[ 1,  0] + omega[ 0,  1]))
    omega = omega.at[ 0, -1].set(0.5 * (omega[ 1, -1] + omega[ 0, -2]))
    omega = omega.at[-1,  0].set(0.5 * (omega[-2,  0] + omega[-1,  1]))
    omega = omega.at[-1, -1].set(0.5 * (omega[-2, -1] + omega[-1, -2]))

    return omega


# ===========================================================================
# Velocity calculation
# ===========================================================================

@partial(jax.jit, static_argnums=(1, 2))
def calculate_velocities(psi: jnp.ndarray, h: float, U: float):
    """Central-difference velocities; apply wall BCs."""
    N = psi.shape[0]

    u = jnp.zeros((N, N))
    v = jnp.zeros((N, N))

    # Interior: u = dψ/dy,  v = -dψ/dx
    u = u.at[1:-1, 1:-1].set((psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h))
    v = v.at[1:-1, 1:-1].set(-(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h))

    # Top lid moves at U
    u = u.at[-1, :].set(U)

    return u, v


# ===========================================================================
# Red-Black SOR — fully vectorised
# ===========================================================================

@partial(jax.jit, static_argnums=(2, 3, 4))
def sor_sweep(psi: jnp.ndarray,
              omega: jnp.ndarray,
              h: float,
              omega_relax: float,
              N: int) -> jnp.ndarray:
    """One full red-black SOR iteration."""
    source = h**2 * omega

    i_idx, j_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    interior = (i_idx > 0) & (i_idx < N - 1) & (j_idx > 0) & (j_idx < N - 1)
    red_mask   = interior & ((i_idx + j_idx) % 2 == 0)
    black_mask = interior & ((i_idx + j_idx) % 2 == 1)

    def gs_update(psi_in, mask):
        neighbor_sum = (jnp.roll(psi_in,  1, 0) + jnp.roll(psi_in, -1, 0) +
                        jnp.roll(psi_in,  1, 1) + jnp.roll(psi_in, -1, 1))
        psi_gs  = 0.25 * (neighbor_sum + source)
        psi_new = jnp.where(mask, (1 - omega_relax) * psi_in + omega_relax * psi_gs, psi_in)
        return psi_new

    psi = gs_update(psi, red_mask)
    psi = gs_update(psi, black_mask)
    return psi


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def solve_streamfunction(psi: jnp.ndarray,
                         omega: jnp.ndarray,
                         h: float,
                         omega_relax: float = 1.8,
                         max_sor_iter: int = 1000,
                         tol: float = 1e-4) -> jnp.ndarray:
    """Run SOR until convergence or max_sor_iter sweeps."""
    N = psi.shape[0]

    def cond(carry):
        psi_in, iteration, max_change = carry
        return (max_change >= tol) & (iteration < max_sor_iter)

    def body(carry):
        psi_in, iteration, _ = carry
        psi_new    = sor_sweep(psi_in, omega, h, omega_relax, N)
        max_change = jnp.max(jnp.abs(psi_new - psi_in))
        return psi_new, iteration + 1, max_change

    psi_out, _, _ = jax.lax.while_loop(cond, body, (psi, 0, jnp.inf))
    return psi_out


# ===========================================================================
# ADI vorticity transport
# ===========================================================================

@partial(jax.jit, static_argnums=(3, 4, 5))
def solve_vorticity_adi(omega: jnp.ndarray,
                        u_c: jnp.ndarray,
                        v_c: jnp.ndarray,
                        nu: float,
                        h: float,
                        dt: float) -> jnp.ndarray:
    """
    One ADI time step for the vorticity transport equation.

    Step 1 : implicit in x (rows),  explicit in y + convection
    Step 2 : implicit in y (cols),  explicit in x* + convection
    """
    N   = omega.shape[0]
    M   = N - 2          # number of interior rows / cols
    alpha = (nu * dt) / (2.0 * h**2)

    # ------------------------------------------------------------------
    # Explicit convection (first-order upwind, computed once at time n)
    # ------------------------------------------------------------------
    ow = omega[1:-1, 1:-1]   # interior vorticity (M, M)

    domega_dx = jnp.where(
        u_c > 0,
        (ow - omega[1:-1,  :-2]) / h,
        (omega[1:-1, 2:] - ow)  / h,
    )
    domega_dy = jnp.where(
        v_c > 0,
        (ow - omega[ :-2, 1:-1]) / h,
        (omega[2:, 1:-1] - ow)   / h,
    )
    conv = u_c * domega_dx + v_c * domega_dy   # (M, M)

    # Explicit diffusion in y (for Step 1 RHS)
    diff_y = (nu / h**2) * (omega[2:, 1:-1] + omega[:-2, 1:-1] - 2.0 * ow)

    # ------------------------------------------------------------------
    # Step 1 : implicit in x → solve M tridiagonal systems of size M
    #          each system corresponds to one row i
    # ------------------------------------------------------------------
    # RHS[i, j] = omega[i,j] + dt*(diff_y - conv)
    RHS1 = ow + dt * (diff_y - conv)            # (M, M)

    # Build tridiagonal (constant coefficients across all rows)
    a1 = jnp.full((M, M), -alpha)
    b1 = jnp.full((M, M),  1.0 + 2.0 * alpha)
    c1 = jnp.full((M, M), -alpha)
    # Zero out sub/super diag at boundaries (they don't exist there)
    a1 = a1.at[:, 0].set(0.0)
    c1 = c1.at[:, -1].set(0.0)

    # Incorporate wall BCs into RHS
    RHS1 = RHS1.at[:, 0 ].add(alpha * omega[1:-1,  0])
    RHS1 = RHS1.at[:, -1].add(alpha * omega[1:-1, -1])

    omega_star_interior = tdma_batch(a1, b1, c1, RHS1)   # (M, M)

    # Assemble full omega_star (boundaries from current omega)
    omega_star = omega.at[1:-1, 1:-1].set(omega_star_interior)

    # ------------------------------------------------------------------
    # Explicit diffusion in x using omega_star (for Step 2 RHS)
    # ------------------------------------------------------------------
    ow_s   = omega_star[1:-1, 1:-1]
    diff_x = (nu / h**2) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * ow_s)

    # RHS Step 2: omega_star + dt*(diff_x* - conv_n)
    RHS2 = ow_s + dt * (diff_x - conv)          # (M, M)

    # ------------------------------------------------------------------
    # Step 2 : implicit in y → solve M tridiagonal systems of size M
    #          each system corresponds to one column j
    #          transpose so that each "row" in the batch = one column
    # ------------------------------------------------------------------
    RHS2_T = RHS2.T   # (M, M)  — now row index = column index j

    a2 = jnp.full((M, M), -alpha)
    b2 = jnp.full((M, M),  1.0 + 2.0 * alpha)
    c2 = jnp.full((M, M), -alpha)
    a2 = a2.at[:, 0].set(0.0)
    c2 = c2.at[:, -1].set(0.0)

    # Incorporate wall BCs
    RHS2_T = RHS2_T.at[:, 0 ].add(alpha * omega_star[ 0, 1:-1])
    RHS2_T = RHS2_T.at[:, -1].add(alpha * omega_star[-1, 1:-1])

    omega_new_interior_T = tdma_batch(a2, b2, c2, RHS2_T)  # (M, M)
    omega_new_interior   = omega_new_interior_T.T            # (M, M)

    omega_new = omega.at[1:-1, 1:-1].set(omega_new_interior)
    return omega_new


# ===========================================================================
# Pressure (Jacobi iteration — post-processing only)
# ===========================================================================

@partial(jax.jit, static_argnums=(1, 2, 3))
def calculate_pressure(uv_fields: tuple,
                       h: float,
                       max_iter: int = 5000,
                       tol: float = 1e-6) -> jnp.ndarray:
    u, v = uv_fields
    N    = u.shape[0]

    dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
    dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)

    rhs       = jnp.zeros((N, N))
    rhs       = rhs.at[1:-1, 1:-1].set(-(dudx**2 + 2.0 * dudy * dvdx + dvdy**2))

    def cond(carry):
        p_in, it, chg = carry
        return (chg >= tol) & (it < max_iter)

    def body(carry):
        p_in, it, _ = carry
        p_new = p_in.at[1:-1, 1:-1].set(
            0.25 * (p_in[2:, 1:-1] + p_in[:-2, 1:-1] +
                    p_in[1:-1, 2:] + p_in[1:-1, :-2] - h**2 * rhs[1:-1, 1:-1])
        )
        chg = jnp.max(jnp.abs(p_new - p_in))
        return p_new, it + 1, chg

    p0 = jnp.zeros((N, N))
    p, _, _ = jax.lax.while_loop(cond, body, (p0, 0, jnp.inf))
    p = p - jnp.mean(p[1:-1, 1:-1])
    return p


# ===========================================================================
# Main solver class
# ===========================================================================

class LidDrivenCavitySolverJAX:
    """
    Pure-JAX lid-driven cavity solver.

    All inner-loop computations run on the device (GPU/TPU/CPU XLA) via JIT.
    Python-level code is limited to orchestration and I/O.
    """

    def __init__(self, N: int = 101, Re: float = 100.0,
                 lid_velocity: float = 1.0, L: float = 1.0):
        self.N   = N
        self.Re  = Re
        self.U   = float(lid_velocity)
        self.L   = float(L)
        self.h   = L / (N - 1)
        self.nu  = self.U * self.L / Re

        # Time step (CFL-based)
        self.dt  = 0.1 * self.h / self.U

        print(f"[INFO] N={N}, Re={Re}, h={self.h:.4e}, dt={self.dt:.4e}, "
              f"nu={self.nu:.4e}")

        # Grid (host-side for plotting)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        self.x_np, self.y_np = x, y
        self.X_np, self.Y_np = np.meshgrid(x, y, indexing='xy')

        # Device-side state
        self.state = CavityState(
            psi   = jnp.zeros((N, N)),
            omega = jnp.zeros((N, N)),
            u     = jnp.zeros((N, N)),
            v     = jnp.zeros((N, N)),
        )
        self.p = jnp.zeros((N, N))

        # Convergence log
        self.history: dict[str, list] = {'iterations': [], 'max_change': [], 'psi_min': []}

    # ------------------------------------------------------------------
    # One full outer iteration
    # ------------------------------------------------------------------

    def _step(self, state: CavityState) -> tuple[CavityState, float]:
        psi, omega, u, v = state

        # 1. Velocities
        u, v = calculate_velocities(psi, self.h, self.U)

        # 2. BCs
        omega = apply_bcs(psi, omega, self.h, self.U)

        # 3. Vorticity transport (ADI)
        u_c = u[1:-1, 1:-1]
        v_c = v[1:-1, 1:-1]
        omega_new = solve_vorticity_adi(omega, u_c, v_c, self.nu, self.h, self.dt)

        # 4. BCs on new omega
        omega_new = apply_bcs(psi, omega_new, self.h, self.U)

        # 5. Streamfunction (SOR)
        psi_new = solve_streamfunction(psi, omega_new, self.h)

        # Convergence metric
        max_change = jnp.max(jnp.abs(omega_new[1:-1, 1:-1] - omega[1:-1, 1:-1]))

        new_state = CavityState(psi=psi_new, omega=omega_new, u=u, v=v)
        return new_state, max_change

    # ------------------------------------------------------------------
    # Full solve loop
    # ------------------------------------------------------------------

    def solve(self, max_iterations: int = 50000, tolerance: float = 1e-6):
        print(f"\n{'='*60}")
        print(f"LID-DRIVEN CAVITY  —  JAX / GPU")
        print(f"{'='*60}")
        print(f"Grid {self.N}×{self.N}   Re={self.Re}   "
              f"dt={self.dt:.2e}   tol={tolerance:.1e}")
        print(f"{'='*60}")

        # Warm up JIT (first call compiles — excluded from timing)
        print("[INFO] Warming up JIT compilation …")
        _ = self._step(self.state)
        jax.block_until_ready(_[0].psi)
        print("[INFO] Compilation done.\n")

        state      = self.state
        converged  = False
        t0         = time.perf_counter()

        for it in range(max_iterations):
            state, max_change = self._step(state)

            # Pull scalar to host only every 5000 iters (avoid transfer overhead)
            if it % 5000 == 0 or it < 10:
                chg_val = float(max_change)
                psi_min = float(jnp.min(state.psi))
                self.history['iterations'].append(it)
                self.history['max_change'].append(chg_val)
                self.history['psi_min'].append(psi_min)
                elapsed = time.perf_counter() - t0
                print(f"  iter {it:6d}  Δω_max={chg_val:.3e}  "
                      f"ψ_min={psi_min:.6f}  t={elapsed:.1f}s")

                if chg_val < tolerance and it > 1000:
                    converged = True
                    break

        elapsed = time.perf_counter() - t0
        jax.block_until_ready(state.psi)

        self.state = state
        if converged:
            print(f"\n[SUCCESS] Converged at iter {it} in {elapsed:.2f}s")
        else:
            print(f"\n[WARNING] Max iterations reached ({elapsed:.2f}s)")

        # Final velocity + pressure
        u, v = calculate_velocities(state.psi, self.h, self.U)
        self.state = CavityState(psi=state.psi, omega=state.omega, u=u, v=v)
        self.p     = calculate_pressure((u, v), self.h)

        return converged, it

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def get_vortex_center(self):
        psi_np = np.array(self.state.psi)
        idx    = np.unravel_index(np.argmin(psi_np), psi_np.shape)
        return self.x_np[idx[1]], self.y_np[idx[0]]

    def print_summary(self):
        vx, vy    = self.get_vortex_center()
        vel_mag   = jnp.sqrt(self.state.u**2 + self.state.v**2)
        print(f"\n{'='*60}")
        print("SOLUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Grid : {self.N}×{self.N}   Re = {self.Re}")
        print(f"Lid  : U = {self.U}  (top wall, y = {self.L})")
        print(f"ψ    : min = {float(jnp.min(self.state.psi)):.6f}")
        print(f"ω    : min = {float(jnp.min(self.state.omega)):.3f}, "
              f"max = {float(jnp.max(self.state.omega)):.3f}")
        print(f"|V|  : max = {float(jnp.max(vel_mag)):.4f}") 
        print(f"p    : min = {float(jnp.min(self.p)):.3f}, "
              f"max = {float(jnp.max(self.p)):.3f}")
        print(f"Vortex center : x = {vx:.4f},  y = {vy:.4f}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def create_plots(self, save_fig: bool = True):
        """Six-panel diagnostic figure."""
        plt.style.use('seaborn-v0_8-whitegrid')

        psi_np   = np.array(self.state.psi)
        omega_np = np.array(self.state.omega)
        u_np     = np.array(self.state.u)
        v_np     = np.array(self.state.v)
        p_np     = np.array(self.p)
        vmag     = np.sqrt(u_np**2 + v_np**2)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=200)
        axs = axes.ravel()

        X, Y = self.X_np, self.Y_np

        # 1 Streamlines
        lev = np.linspace(psi_np.min(), 0, 21)
        axs[0].contourf(X, Y, psi_np, levels=lev, cmap='viridis', alpha=0.6)
        axs[0].contour( X, Y, psi_np, levels=lev, colors='#2E86AB', linewidths=0.8)
        axs[0].set_title('Streamlines ψ',  fontweight='bold')
        axs[0].set_aspect('equal'); axs[0].grid(False)

        # 2 Vorticity
        cf2 = axs[1].contourf(X, Y, omega_np, levels=40, cmap='coolwarm', alpha=0.9)
        fig.colorbar(cf2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title('Vorticity ω', fontweight='bold')
        axs[1].set_aspect('equal')

        # 3 Velocity magnitude
        cf3 = axs[2].contourf(X, Y, vmag, levels=30, cmap='plasma', alpha=0.9)
        fig.colorbar(cf3, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].set_title('|V|', fontweight='bold')
        axs[2].set_aspect('equal')

        # 4 u at vertical centreline
        xi = np.argmin(np.abs(self.x_np - self.L / 2))
        axs[3].plot(u_np[:, xi], self.y_np, 'b-o', markersize=3, linewidth=2,
                    label=f'u at x={self.L/2:.2f}')
        axs[3].set_xlabel('u-velocity'); axs[3].set_ylabel('y')
        axs[3].set_title('u-Velocity (vertical CL)', fontweight='bold')
        axs[3].legend(); axs[3].grid(True, alpha=0.4, ls='--')

        # 5 v at horizontal centreline
        yi = np.argmin(np.abs(self.y_np - self.L / 2))
        axs[4].plot(self.x_np, v_np[yi, :], 'r-s', markersize=3, linewidth=2,
                    label=f'v at y={self.L/2:.2f}')
        axs[4].set_xlabel('x'); axs[4].set_ylabel('v-velocity')
        axs[4].set_title('v-Velocity (horizontal CL)', fontweight='bold')
        axs[4].legend(); axs[4].grid(True, alpha=0.4, ls='--')

        # 6 Pressure
        cf6 = axs[5].contourf(X, Y, p_np, levels=30, cmap='RdGy', alpha=0.9)
        fig.colorbar(cf6, ax=axs[5], fraction=0.046, pad=0.04)
        axs[5].set_title('Pressure', fontweight='bold')
        axs[5].set_aspect('equal')

        for ax in [axs[0], axs[1], axs[2], axs[5]]:
            ax.set_xlim(0, self.L); ax.set_ylim(0, self.L)

        plt.suptitle(
            f'Lid-Driven Cavity — Re={self.Re}, Grid={self.N}×{self.N}\n'
            f'(JAX / {jax.default_backend().upper()})',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        if save_fig:
            fname = f'lid_driven_jax_Re{self.Re}_N{self.N}.png'
            plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Saved: {fname}")
        plt.show()

        # Convergence
        if len(self.history['iterations']) > 1:
            fig2, ax = plt.subplots(figsize=(9, 5), dpi=150)
            ax.semilogy(self.history['iterations'], self.history['max_change'],
                        'b-', linewidth=2, label='max Δω')
            ax.set_xlabel('Iteration'); ax.set_ylabel('Max change')
            ax.set_title('Convergence History', fontweight='bold')
            ax.grid(True, alpha=0.4, ls='--'); ax.legend()
            if save_fig:
                fn2 = f'convergence_jax_Re{self.Re}_N{self.N}.png'
                plt.savefig(fn2, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"[SUCCESS] Saved: {fn2}")
            plt.show()

    def create_showcase_plots(self, save_fig: bool = True):
        """Minimal two-panel showcase (streamlines + centreline u)."""
        plt.style.use('seaborn-v0_8-white')

        psi_np = np.array(self.state.psi)
        u_np   = np.array(self.state.u)
        X, Y   = self.X_np, self.Y_np

        fig, axs = plt.subplots(1, 2, figsize=(18, 8), dpi=200)

        lev = np.linspace(psi_np.min(), 0, 25)
        axs[0].contourf(X, Y, psi_np, levels=lev, cmap='viridis', alpha=0.85)
        axs[0].contour( X, Y, psi_np, levels=lev, colors='black', linewidths=1.2, alpha=0.55)
        axs[0].set_title('Streamlines (ψ)', fontsize=15, fontweight='bold')
        axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
        axs[0].set_aspect('equal'); axs[0].grid(False)

        xi = np.argmin(np.abs(self.x_np - self.L / 2))
        axs[1].plot(u_np[:, xi], self.y_np, 'k-', linewidth=3,
                    label=r'$u(y)$ at $x = L/2$')
        axs[1].set_title('Centerline Velocity Profile', fontsize=15, fontweight='bold')
        axs[1].set_xlabel('u-velocity'); axs[1].set_ylabel('y')
        axs[1].grid(True, alpha=0.3, ls='--'); axs[1].legend(fontsize=11)

        plt.suptitle(
            f'Lid-Driven Cavity — Re={self.Re}, Grid={self.N}×{self.N}  '
            f'[JAX · {jax.default_backend().upper()}]',
            fontsize=17, fontweight='bold', y=1.0
        )
        plt.tight_layout()

        if save_fig:
            fname = f'showcase_jax_Re{self.Re}_N{self.N}.png'
            plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Saved: {fname}")
        plt.show()

    def save_arrays(self, save_dir='.'):
        """Save solution fields as compressed NumPy archive."""
        fname = f'{save_dir}/flow_fields_jax_Re{int(self.Re)}_N{self.N}.npz'
        np.savez_compressed(
            fname,
            x     = self.x_np,
            y     = self.y_np,
            psi   = np.array(self.state.psi),
            omega = np.array(self.state.omega),
            u     = np.array(self.state.u),
            v     = np.array(self.state.v),
            p     = np.array(self.p),
        )
        print(f"[SUCCESS] Arrays saved: {fname}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='JAX Lid-Driven Cavity Solver')
    p.add_argument('--N',       type=int,   default=101,   help='Grid size NxN')
    p.add_argument('--Re',      type=float, default=100.0, help='Reynolds number')
    p.add_argument('--maxiter', type=int,   default=50000, help='Max outer iterations')
    p.add_argument('--tol',     type=float, default=1e-6,  help='Convergence tolerance')
    p.add_argument('--U',       type=float, default=1.0,   help='Lid velocity')
    p.add_argument('--no-save-figure', action='store_true',        help='Skip saving figures')
    p.add_argument('--save-dir', type=str,   default='./results', help='Directory to save arrays')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    
    Re_range = [50, 1000]
    Re_range_log = [np.log(Re_range[0]), np.log(Re_range[1])]
    # uniformly sample 256 Re values in log space
    np.random.seed(42)
    Re_values_log = np.random.uniform(Re_range_log[0], Re_range_log[1], 256,)
    Re_values = np.int64(np.exp(Re_values_log))
    print(f"Randomly sampled Re values (log-uniform): {Re_values}")
    
    for idx, Re in enumerate(Re_values):
        print(f"Num {idx} of {len(Re_values)}: Re = {Re}")
        solver = LidDrivenCavitySolverJAX(
            N            = args.N,
            Re           = Re,
            lid_velocity = args.U,
        )

        converged, iters = solver.solve(
            max_iterations = args.maxiter,
            tolerance      = args.tol,
        )

        solver.print_summary()
        solver.save_arrays(save_dir=args.save_dir)