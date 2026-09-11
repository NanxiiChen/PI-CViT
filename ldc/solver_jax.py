"""
Lid-Driven Cavity Flow Solver (2D, Incompressible) — Pure JAX / GPU Implementation
====================================================================================

Numerical methods:
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport (implicit diffusion + explicit upwind convection;
  the constant-coefficient tridiagonal solves are done via a precomputed inverse)
- Red–Black SOR for the streamfunction Poisson equation (fixed sweep count)

Performance design:
  • the whole outer iteration is fused into ONE jitted function — no per-call
    kernel dispatch overhead inside the loop
  • outer iterations run in lax.fori_loop blocks inside a single XLA computation,
    with host-side convergence checks only between blocks
  • all static masks / matrices are precomputed once at initialisation

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

    JAX_ENABLE_X64=0 python lid_driven_cavity_jax.py   # float32 (faster, lower accuracy)
    python lid_driven_cavity_jax.py --float32          # same as above

Requirements
-----------
    pip install jax[cuda12]   # for GPU  (or jax[cpu] for CPU-only)
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Float precision: float64 by default (original numerics — the smooth lid's
# sharp corner profile stalls at ~1e-2 in float32 on coarse grids).
# On this workload the kernels are latency-bound, so x64 costs only ~15%.
# Set JAX_ENABLE_X64=0 (or pass --float32) to opt into float32.
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")

print(f"JAX version : {jax.__version__}")
print(f"Backend     : {jax.default_backend()}")
print(f"Devices     : {jax.devices()}")
print(f"JAX x64     : {jax.config.x64_enabled}")


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
# Boundary conditions
# ===========================================================================

def apply_bcs(psi: jnp.ndarray,
              omega: jnp.ndarray,
              h: float,
              lid_u: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Thom's formula vorticity BCs; return updated omega.
    psi boundaries are already zero and stay zero — no update needed.
    Top lid velocity follows the smooth profile
    lid_u(x) = U * (1 - cosh(50(x - 0.5)) / cosh(25))  → 0 at corners.
    """
    # Top wall (moving lid, smooth profile)
    omega_top    = -2.0 * psi[-2, 1:-1] / h**2 - 2.0 * lid_u[1:-1] / h
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

def calculate_velocities(psi: jnp.ndarray, h: float, lid_u: jnp.ndarray):
    """Central-difference velocities; apply wall BCs."""
    N = psi.shape[0]

    u = jnp.zeros((N, N))
    v = jnp.zeros((N, N))

    # Interior: u = dψ/dy,  v = -dψ/dx
    u = u.at[1:-1, 1:-1].set((psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h))
    v = v.at[1:-1, 1:-1].set(-(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h))

    # Top lid moves with smooth profile lid_u(x)  (zero at corners)
    u = u.at[-1, :].set(lid_u)

    return u, v


# ===========================================================================
# Red-Black SOR — fixed sweep count, fully vectorised
# ===========================================================================

def _gs_pass(psi: jnp.ndarray,
             source: jnp.ndarray,
             mask: jnp.ndarray,
             omega_relax: float) -> jnp.ndarray:
    neighbor_sum = (jnp.roll(psi,  1, 0) + jnp.roll(psi, -1, 0) +
                    jnp.roll(psi,  1, 1) + jnp.roll(psi, -1, 1))
    psi_gs  = 0.25 * (neighbor_sum + source)
    return jnp.where(mask, (1 - omega_relax) * psi + omega_relax * psi_gs, psi)


def sor_sweeps(psi: jnp.ndarray,
               omega: jnp.ndarray,
               h: float,
               omega_relax: float,
               n_sor: int,
               red_mask: jnp.ndarray,
               black_mask: jnp.ndarray) -> jnp.ndarray:
    """
    n_sor red-black SOR sweeps (fixed count — no host syncs per sweep).
    psi starts from the previous outer iteration, so a handful of sweeps
    is enough to keep the Poisson residual small.
    """
    source = h**2 * omega

    def sweep(_i, psi_in):
        psi_in = _gs_pass(psi_in, source, red_mask,   omega_relax)
        psi_in = _gs_pass(psi_in, source, black_mask, omega_relax)
        return psi_in

    return jax.lax.fori_loop(0, n_sor, sweep, psi)


# ===========================================================================
# ADI vorticity transport
# ===========================================================================

def adi_transport(omega: jnp.ndarray,
                  u_c: jnp.ndarray,
                  v_c: jnp.ndarray,
                  A_inv: jnp.ndarray,
                  nu: float,
                  h: float,
                  dt: float) -> jnp.ndarray:
    """
    One ADI time step for the vorticity transport equation.

    Step 1 : implicit in x (rows),  explicit in y + convection
    Step 2 : implicit in y (cols),  explicit in x* + convection

    The implicit diffusion matrix is the same constant-coefficient
    tridiagonal system for every row / column, so both linear solves
    reduce to two dense matmuls with the precomputed inverse A_inv
    (no sequential Thomas sweep on the device).
    """
    M     = omega.shape[0] - 2
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
    # Step 1 : implicit in x → each row i solves  A x_i = RHS1[i, :]
    # ------------------------------------------------------------------
    RHS1 = ow + dt * (diff_y - conv)            # (M, M)
    RHS1 = RHS1.at[:, 0 ].add(alpha * omega[1:-1,  0])
    RHS1 = RHS1.at[:, -1].add(alpha * omega[1:-1, -1])
    omega_star_interior = RHS1 @ A_inv          # (M, M), A is symmetric

    # Assemble full omega_star (boundaries from current omega)
    omega_star = omega.at[1:-1, 1:-1].set(omega_star_interior)

    # ------------------------------------------------------------------
    # Explicit diffusion in x using omega_star (for Step 2 RHS)
    # ------------------------------------------------------------------
    ow_s   = omega_star[1:-1, 1:-1]
    diff_x = (nu / h**2) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * ow_s)

    # ------------------------------------------------------------------
    # Step 2 : implicit in y → each column j solves  A y_j = RHS2[:, j]
    # ------------------------------------------------------------------
    RHS2 = ow_s + dt * (diff_x - conv)          # (M, M)
    RHS2 = RHS2.at[ 0, :].add(alpha * omega_star[ 0, 1:-1])
    RHS2 = RHS2.at[-1, :].add(alpha * omega_star[-1, 1:-1])
    omega_new_interior = A_inv @ RHS2           # (M, M)

    return omega.at[1:-1, 1:-1].set(omega_new_interior)


# ===========================================================================
# Fused outer step + block-scan runner
# ===========================================================================

def step_impl(state: CavityState,
              lid_u: jnp.ndarray,
              red_mask: jnp.ndarray,
              black_mask: jnp.ndarray,
              A_inv: jnp.ndarray,
              h: float,
              nu: float,
              dt: float,
              n_sor: int,
              omega_relax: float) -> tuple[CavityState, jnp.ndarray]:
    """
    One full outer iteration (plain function — fused into run_block by XLA).
    """
    psi, omega, u, v = state

    # 1. Velocities
    u, v = calculate_velocities(psi, h, lid_u)

    # 2. BCs
    omega = apply_bcs(psi, omega, h, lid_u)

    # 3. Vorticity transport (ADI)
    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]
    omega_new = adi_transport(omega, u_c, v_c, A_inv, nu, h, dt)

    # 4. BCs on new omega
    omega_new = apply_bcs(psi, omega_new, h, lid_u)

    # 5. Streamfunction (fixed number of SOR sweeps)
    psi_new = sor_sweeps(psi, omega_new, h, omega_relax, n_sor,
                         red_mask, black_mask)

    # Convergence metric
    max_change = jnp.max(jnp.abs(omega_new[1:-1, 1:-1] - omega[1:-1, 1:-1]))

    new_state = CavityState(psi=psi_new, omega=omega_new, u=u, v=v)
    return new_state, max_change


@partial(jax.jit, static_argnums=(10,))
def run_block(state: CavityState,
              lid_u: jnp.ndarray,
              red_mask: jnp.ndarray,
              black_mask: jnp.ndarray,
              A_inv: jnp.ndarray,
              h: float,
              nu: float,
              dt: float,
              n_sor: int,
              omega_relax: float,
              block_len: int) -> tuple[CavityState, jnp.ndarray]:
    """
    Run block_len fused outer iterations inside a single XLA computation
    (lax.fori_loop — no host syncs, no per-iteration dispatch).
    Returns the new state and the max Δω over the whole block.
    """
    def body(_i, carry):
        st, mx = carry
        new_st, chg = step_impl(st, lid_u, red_mask, black_mask, A_inv,
                                h, nu, dt, n_sor, omega_relax)
        return new_st, jnp.maximum(mx, chg)

    return jax.lax.fori_loop(0, block_len, body, (state, 0.0))


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
                 lid_velocity: float = 1.0, L: float = 1.0,
                 smooth_lid: bool = True, n_sor: int = 10,
                 omega_relax: float = 1.8):
        self.N   = N
        self.Re  = Re
        self.U   = float(lid_velocity)
        self.L   = float(L)
        self.h   = L / (N - 1)
        self.nu  = self.U * self.L / Re
        self.smooth_lid  = smooth_lid
        self.n_sor       = n_sor
        self.omega_relax = omega_relax

        # Time step (CFL-based)
        self.dt = 0.1 * self.h / self.U

        print(f"[INFO] N={N}, Re={Re}, h={self.h:.4e}, dt={self.dt:.4e}, "
              f"nu={self.nu:.4e}, n_sor={n_sor}")

        # Grid (host-side for plotting)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        self.x_np, self.y_np = x, y
        self.X_np, self.Y_np = np.meshgrid(x, y, indexing='xy')

        # Lid velocity profile:
        #   smooth: v1(x, 1) = U * (1 - cosh(50(x - 0.5)) / cosh(25))  (0 at corners)
        #   const : v1(x, 1) = U  (discontinuous at top corners)
        if smooth_lid:
            self.lid_u = jnp.asarray(
                self.U * (1.0 - np.cosh(50.0 * (x / L - 0.5)) / np.cosh(25.0))
            )
            print("[INFO] Lid BC: smooth profile "
                  "U·(1 - cosh(50(x-0.5))/cosh(25))")
        else:
            self.lid_u = jnp.full((N,), self.U)
            print("[INFO] Lid BC: constant U (discontinuous at corners)")

        # Static red-black masks (depend only on N — computed once)
        i_idx, j_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
        interior = (i_idx > 0) & (i_idx < N - 1) & (j_idx > 0) & (j_idx < N - 1)
        self.red_mask   = interior & ((i_idx + j_idx) % 2 == 0)
        self.black_mask = interior & ((i_idx + j_idx) % 2 == 1)

        # Precomputed inverse of the constant-coefficient ADI tridiagonal
        # system (same matrix for every row / column of the interior)
        M     = N - 2
        alpha = self.nu * self.dt / (2.0 * self.h**2)
        a = np.full(M, -alpha)
        b = np.full(M,  1.0 + 2.0 * alpha)
        c = np.full(M, -alpha)
        a[0] = 0.0
        c[-1] = 0.0
        A = np.diag(a[1:], -1) + np.diag(b, 0) + np.diag(c[:-1], 1)
        A_inv = np.linalg.inv(A)
        self.A_inv = jnp.asarray(A_inv).astype(jnp.zeros((1,)).dtype)

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
    # Full solve loop (outer iterations in fused JIT blocks)
    # ------------------------------------------------------------------

    def solve(self, max_iterations: int = 50000, tolerance: float = 1e-6,
              block_len: int = 5000):
        print(f"\n{'='*60}")
        print(f"LID-DRIVEN CAVITY  —  JAX / GPU")
        print(f"{'='*60}")
        print(f"Grid {self.N}×{self.N}   Re={self.Re}   "
              f"dt={self.dt:.2e}   tol={tolerance:.1e}   "
              f"block={block_len}   n_sor={self.n_sor}")
        print(f"{'='*60}")

        args = (self.lid_u, self.red_mask, self.black_mask, self.A_inv,
                self.h, self.nu, self.dt, self.n_sor, self.omega_relax)

        # Warm up JIT (first call compiles — excluded from timing)
        print("[INFO] Warming up JIT compilation …")
        _ = run_block(self.state, *args, block_len)
        jax.block_until_ready(_[0].psi)
        print("[INFO] Compilation done.\n")

        state      = self.state
        converged  = False
        it         = 0
        t0         = time.perf_counter()

        while it < max_iterations:
            nb = min(block_len, max_iterations - it)
            state, block_max = run_block(state, *args, nb)
            it += nb

            # Pull scalar to host only once per block (amortised overhead)
            chg_val = float(block_max)
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
        u, v = calculate_velocities(state.psi, self.h, self.lid_u)
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
        if self.smooth_lid:
            print(f"Lid  : smooth profile U·(1 - cosh(50(x-0.5))/cosh(25)), "
                  f"U = {self.U}  (top wall, y = {self.L})")
        else:
            print(f"Lid  : constant U = {self.U}  (top wall, y = {self.L})")
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

    def save_arrays(self, save_id, save_dir='.'):
        """Save solution fields as compressed NumPy archive."""
        fname = f'{save_dir}/flow_fields_jax_N{self.N}_{save_id}.npz'
        np.savez_compressed(
            fname,
            x     = self.x_np,
            y     = self.y_np,
            psi   = np.array(self.state.psi),
            omega = np.array(self.state.omega),
            u     = np.array(self.state.u),
            v     = np.array(self.state.v),
            p     = np.array(self.p),
            re    = self.Re
        )
        print(f"[SUCCESS] Arrays saved: {fname}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='JAX Lid-Driven Cavity Solver')
    p.add_argument('--N',       type=int,   default=256,   help='Grid size NxN')
    p.add_argument('--Re',      type=float, default=100.0, help='Reynolds number')
    p.add_argument('--maxiter', type=int,   default=100000, help='Max outer iterations')
    p.add_argument('--tol',     type=float, default=1e-6,  help='Convergence tolerance')
    p.add_argument('--U',       type=float, default=1.0,   help='Lid velocity')
    p.add_argument('--n-sor',   type=int,   default=20,    help='SOR sweeps per outer iteration')
    p.add_argument('--block-len', type=int, default=5000,  help='Fused iterations per JIT block')
    p.add_argument('--float32', action='store_true',
                   help='Use float32 (faster; smooth-lid corners may stall ~1e-2)')
    p.add_argument('--no-smooth-lid', action='store_true',
                   help='Use constant lid velocity (discontinuous at corners)')
    p.add_argument('--no-save-figure', action='store_true',        help='Skip saving figures')
    p.add_argument('--save-dir', type=str,   default='./results', help='Directory to save arrays')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.float32:
        jax.config.update("jax_enable_x64", False)
        print(f"JAX x64     : {jax.config.x64_enabled}")
    
    
    Re_range = [50, 1000]
    Re_range_log = [np.log(Re_range[0]), np.log(Re_range[1])]
    # uniformly sample 100 Re values in log space
    np.random.seed(23)
    Re_values_log = np.random.uniform(Re_range_log[0], Re_range_log[1], 100,)
    Re_values = np.exp(Re_values_log)
    print(f"Randomly sampled Re values (log-uniform): {Re_values}")
    np.save(f'{args.save_dir}/Re_values.npy', Re_values)
    
    # Re_values = np.array([50, 100, 200, 500, 1000])
    # np.save(f'{args.save_dir}/Re_values.npy', Re_values)
    
    for idx, Re in enumerate(Re_values):
        print(f"Num {idx} of {len(Re_values)}: Re = {Re}")
        solver = LidDrivenCavitySolverJAX(
            N            = args.N,
            Re           = Re,
            lid_velocity = args.U,
            smooth_lid   = not args.no_smooth_lid,
            n_sor        = args.n_sor,
        )

        converged, iters = solver.solve(
            max_iterations = args.maxiter,
            tolerance      = args.tol,
            block_len      = args.block_len,
        )

        solver.print_summary()
        # solver.create_plots(save_fig=not args.no_save_figure)
        # solver.create_showcase_plots(save_fig=not args.no_save_figure)
        solver.save_arrays(save_id=idx, save_dir=args.save_dir)