#LID-DRIVEN CAVITY FLOW SOLVER
"""
Lid-Driven Cavity Flow Solver (2D, Incompressible)

Numerical methods:
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport
- Red–Black SOR for streamfunction Poisson equation

Author: Kartikey Singh
Year: 2026
License: MIT

This code is intended as an educational and reference implementation.
It does not claim methodological novelty.

This implementation emphasizes numerical clarity,
physical correctness, and reproducibility over performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sys
import io

# ---- NumPy pickle compatibility patch ----
# Required for loading pickled NumPy objects across some environments
# (e.g., Colab / Kaggle kernel differences)
sys.modules['numpy._core.numeric'] = np.core.numeric

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

class LidDrivenCavitySolver:
    """
    Solves 2D incompressible lid-driven cavity flow using the psi-omega method.
    (Uses Red-Black SOR for psi and the ADI method for omega transport.)
    """
    
    def __init__(self, N=51, Re=100, lid_velocity=1.0, L=1.0):
        """
        Initialize with CORRECT coordinate system.
        """
        self.N = N
        self.Re = Re
        self.U = lid_velocity
        self.L = L
        
        # Grid parameters
        self.h = L / (N - 1)
        self.nu = self.U * self.L / self.Re # Kinematic viscosity
        
        # Grid coordinates
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy') 
        
        # Flow variables
        self.psi = np.zeros((N, N))
        self.omega = np.zeros((N, N))
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.p = np.zeros((N, N))
        
        # Time step (ADI eliminates the diffusion stability constraint.)
        # Reduced CFL safety factor to improve stability of explicit upwind convection
        CFL_SAFETY_FACTOR = 0.1 
        dt_cfl = CFL_SAFETY_FACTOR * self.h / self.U
        self.dt = dt_cfl
        print(f"[INFO] Time step selected (CFL-based): dt = {self.dt:.2e} (nu={self.nu:.2e}, h={self.h:.2e})")
        
        # Convergence tracking
        self.history = {'iterations': [], 'max_change': [], 'psi_min': []}
        
        # ADI parameters (stored for reuse in ADI solver)
        self.alpha_adi = (self.nu * self.dt) / (2 * self.h**2)
        # Arrays for intermediate velocity components used in convection terms
        self.u_c = np.zeros((N-2, N-2))
        self.v_c = np.zeros((N-2, N-2))
    
    def apply_boundary_conditions(self):
        """Apply BCs with CORRECT wall identification"""
        N = self.N
        h = self.h
        U = self.U
        
        # Streamfunction BCs (all walls are no-slip, psi=0)
        self.psi[0, :] = 0
        self.psi[-1, :] = 0
        self.psi[:, 0] = 0
        self.psi[:, -1] = 0
        
        # Vorticity BCs (Thom's formula) - based on current psi
        # Top wall (moving lid, i = N-1)
        self.omega[-1, 1:-1] = -2.0 * self.psi[-2, 1:-1] / h**2 - 2.0 * U / h
        
        # Bottom wall (i = 0)
        self.omega[0, 1:-1] = -2.0 * self.psi[1, 1:-1] / h**2
        
        # Left wall (j = 0)
        self.omega[1:-1, 0] = -2.0 * self.psi[1:-1, 1] / h**2
        
        # Right wall (j = N-1)
        self.omega[1:-1, -1] = -2.0 * self.psi[1:-1, -2] / h**2
        
        # Corner vorticity (Averaging adjacent wall points)
        self.omega[0, 0] = 0.5 * (self.omega[1, 0] + self.omega[0, 1])
        self.omega[0, -1] = 0.5 * (self.omega[1, -1] + self.omega[0, -2])
        self.omega[-1, 0] = 0.5 * (self.omega[-2, 0] + self.omega[-1, 1])
        self.omega[-1, -1] = 0.5 * (self.omega[-2, -1] + self.omega[-1, -2])
    
    def solve_streamfunction(self, max_iterations=1000, tolerance=1e-5, omega_relaxation=1.8):
        """
        Solve Poisson equation for streamfunction using Red-Black SOR (Optimized Vectorized).
        """
        N = self.N
        h = self.h
        
        # Create masks (Done once in initialization normally, but kept here for completeness)
        i_vals, j_vals = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        red_interior_mask = ((i_vals + j_vals) % 2 == 0) & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)
        black_interior_mask = ~red_interior_mask & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)
        
        source_term = h**2 * self.omega
        
        for iteration in range(max_iterations):
            psi_old = self.psi.copy()
            
            # --- Red Sweep ---
            psi_new_red_calc = 0.25 * (
                np.roll(self.psi, 1, axis=0)[red_interior_mask] + 
                np.roll(self.psi, -1, axis=0)[red_interior_mask] + 
                np.roll(self.psi, 1, axis=1)[red_interior_mask] + 
                np.roll(self.psi, -1, axis=1)[red_interior_mask] + 
                source_term[red_interior_mask]
            )
            self.psi[red_interior_mask] = (1 - omega_relaxation) * self.psi[red_interior_mask] + omega_relaxation * psi_new_red_calc

            # --- Black Sweep ---
            psi_new_black_calc = 0.25 * (
                np.roll(self.psi, 1, axis=0)[black_interior_mask] + 
                np.roll(self.psi, -1, axis=0)[black_interior_mask] + 
                np.roll(self.psi, 1, axis=1)[black_interior_mask] + 
                np.roll(self.psi, -1, axis=1)[black_interior_mask] + 
                source_term[black_interior_mask]
            )
            self.psi[black_interior_mask] = (1 - omega_relaxation) * self.psi[black_interior_mask] + omega_relaxation * psi_new_black_calc

            # Check for convergence
            max_change = np.max(np.abs(self.psi - psi_old))
            if max_change < tolerance:
                break
        
        # Return max change for tracking
        return max_change
    
    def calculate_velocities(self):
        """Calculate velocities using Central Differences (interior) and apply BCs."""
        N = self.N
        h = self.h
        
        # Interior points
        # u = dpsi/dy
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[0:-2, 1:-1]) / (2 * h)
        # v = -dpsi/dx
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, 0:-2]) / (2 * h)
        
        # Store interior velocities for ADI solver (u_c and v_c)
        self.u_c = self.u[1:-1, 1:-1]
        self.v_c = self.v[1:-1, 1:-1]
        
        # Boundary conditions
        self.u[-1, :] = self.U    # Top lid
        self.v[-1, :] = 0
        self.u[0, :] = 0          # Bottom, Left, Right walls are static
        self.v[0, :] = 0
        self.u[:, 0] = 0
        self.v[:, 0] = 0
        self.u[:, -1] = 0
        self.v[:, -1] = 0

    def solve_vorticity_transport_ADI(self):
        """
        Solve vorticity transport equation using the ADI method (Implicit-Explicit).
        The implicit part is for the diffusion terms. Convection is treated explicitly
        using the Upwind scheme for stability.
        
        Uses second-order central difference for diffusion (implicit) and first-order 
        upwind for convection (explicit).
        """
        N = self.N
        h = self.h
        nu = self.nu
        dt = self.dt
        alpha = self.alpha_adi # Pre-calculated alpha for ADI

        omega_old = self.omega.copy()
        
        # --- Step 1: Implicit in X (Diffusion X), Explicit in Y (Diffusion Y + Convection) ---
        # Solve for intermediate field omega_star
        omega_star = np.zeros_like(self.omega)
        omega_star[0, :] = self.omega[0, :] # Apply BCs to star field
        omega_star[-1, :] = self.omega[-1, :]
        omega_star[:, 0] = self.omega[:, 0]
        omega_star[:, -1] = self.omega[:, -1]
        
        # Pre-calculate explicit terms (Diffusion_y and Convection) for RHS of Step 1
        # Convection Terms (Upwind Differencing, explicit)
        # d(omega)/dx term (requires u_center)
        domega_dx_up = np.where(
            self.u_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[1:-1, :-2]) / h, # Backward difference
            (omega_old[1:-1, 2:] - omega_old[1:-1, 1:-1]) / h  # Forward difference
        )
        # d(omega)/dy term (requires v_center)
        domega_dy_up = np.where(
            self.v_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[:-2, 1:-1]) / h, # Backward difference
            (omega_old[2:, 1:-1] - omega_old[1:-1, 1:-1]) / h  # Forward difference
        )
        convection_explicit = self.u_c * domega_dx_up + self.v_c * domega_dy_up # Shape: (N-2, N-2)
        
        # Diffusion Y-Term (Explicit part for Step 1)
        diff_y_explicit = (nu / h**2) * (
            omega_old[2:, 1:-1] + omega_old[:-2, 1:-1] - 2 * omega_old[1:-1, 1:-1]
        ) # Shape: (N-2, N-2)

        # Iterate over y-index (rows) to solve tridiagonal systems (implicit in x)
        for i in range(1, N - 1): 
            # Coefficients for the tridiagonal matrix A*omega_star[i-1, j] + B*omega_star[i, j] + C*omega_star[i+1, j] = D
            A_x = -alpha
            B_x = 1 + 2 * alpha
            C_x = -alpha
            
            # RHS for the tridiagonal solve (contains old omega and explicit terms)
            # RHS[j] = omega_old[i, j] + dt/2 * (Diff_y + Convection)
            RHS_x = omega_old[i, 1:-1] + dt * (diff_y_explicit[i-1, :] - convection_explicit[i-1, :])
            
            # Solve tridiagonal system for omega_star[i, 1:-1] using Thomas Algorithm (TDMA)
            # Forward elimination
            P = np.zeros(N - 2)
            Q = np.zeros(N - 2)
            P[0] = C_x / B_x
            Q[0] = RHS_x[0] / B_x
            for j in range(1, N - 2):
                denom = B_x - A_x * P[j-1]
                P[j] = C_x / denom
                Q[j] = (RHS_x[j] - A_x * Q[j-1]) / denom
            
            # Back substitution
            omega_star[i, -2] = Q[N - 3] # j = N-2
            for j in range(N - 4, -1, -1): # j = N-3 down to 1
                omega_star[i, j+1] = Q[j] - P[j] * omega_star[i, j+2]
        
        # --- Step 2: Implicit in Y (Diffusion Y), Explicit in X (Diffusion X + Convection) ---
        # Solve for new field omega_new using omega_star
        omega_new = np.zeros_like(self.omega)
        omega_new[0, :] = self.omega[0, :] # Apply BCs to new field
        omega_new[-1, :] = self.omega[-1, :]
        omega_new[:, 0] = self.omega[:, 0]
        omega_new[:, -1] = self.omega[:, -1]
        
        # Diffusion X-Term (Explicit part for Step 2, using omega_star)
        diff_x_explicit_star = (nu / h**2) * (
            omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2 * omega_star[1:-1, 1:-1]
        ) # Shape: (N-2, N-2)
        # Note: Convection is typically kept explicit from time 'n' to avoid re-calculating velocities
        
        # Iterate over x-index (columns) to solve tridiagonal systems (implicit in y)
        for j in range(1, N - 1): 
            # Coefficients for the tridiagonal matrix
            A_y = -alpha
            B_y = 1 + 2 * alpha
            C_y = -alpha
            
            # RHS for the tridiagonal solve (contains omega_star and explicit terms)
            # RHS[i] = omega_star[i, j] + dt/2 * (Diff_x* + Convection_n)
            RHS_y = omega_star[1:-1, j] + dt * (diff_x_explicit_star[:, j-1] - convection_explicit[:, j-1])
            
            # Solve tridiagonal system for omega_new[1:-1, j] using Thomas Algorithm (TDMA)
            # Forward elimination
            P = np.zeros(N - 2)
            Q = np.zeros(N - 2)
            P[0] = C_y / B_y
            Q[0] = RHS_y[0] / B_y
            for i in range(1, N - 2):
                denom = B_y - A_y * P[i-1]
                P[i] = C_y / denom
                Q[i] = (RHS_y[i] - A_y * Q[i-1]) / denom
            
            # Back substitution
            omega_new[-2, j] = Q[N - 3] # i = N-2
            for i in range(N - 4, -1, -1): # i = N-3 down to 1
                omega_new[i+1, j] = Q[i] - P[i] * omega_new[i+2, j]

        # Only interior points were solved. The boundaries must be reapplied via BCs later.
        return omega_new
    
    def solve(self, max_iterations=50000, tolerance=1e-6):
        """Main solver loop using ADI for omega and RB-SOR for psi."""
        print(f"\n{'='*60}")
        print(f"LID-DRIVEN CAVITY - ADI & RB-SOR")
        print(f"{'='*60}")
        print(f"Grid: {self.N}x{self.N}, Re = {self.Re}, dt: {self.dt:.2e}")
        print(f"{'='*60}")
        
        start_time = time.time()
        omega_rb = 1.8 
        psi_tol = 1e-4 # Loosen psi tolerance slightly for speed at high Re
        
        prev_max_change = float('inf')

        for iteration in range(max_iterations):
            omega_old = self.omega.copy()
            
            # 1. Calculate Velocities (needed for convection terms in ADI)
            self.calculate_velocities()
            
            # 2. Apply Omega BCs (Needed for ADI step)
            self.apply_boundary_conditions()
            
            # 3. Solve Vorticity Transport (ADI)
            self.omega = self.solve_vorticity_transport_ADI()
            
            # 4. Apply Omega BCs again (after ADI update)
            self.apply_boundary_conditions()
            
            # 5. Solve Streamfunction (RB-SOR)
            psi_max_change = self.solve_streamfunction(max_iterations=1000, tolerance=psi_tol, omega_relaxation=omega_rb)
            
            # 6. Convergence Check
            max_omega_change = np.max(np.abs(self.omega[1:-1, 1:-1] - omega_old[1:-1, 1:-1]))
            total_change_metric = max_omega_change
            
            if total_change_metric < tolerance and iteration > 1000:
                elapsed_time = time.time() - start_time
                print(f"\n[SUCCESS] Converged in {iteration} iterations")
                print(f"Time: {elapsed_time:.2f} seconds")
                print(f"Final Max Omega Change: {total_change_metric:.6e}")
                
                self.calculate_velocities()
                self.calculate_pressure()
                return True, iteration
            
            # Log and print progress
            if iteration % 5000 == 0:
                self.history['iterations'].append(iteration)
                self.history['max_change'].append(total_change_metric)
                self.history['psi_min'].append(self.psi.min())
                
                print(f"Iteration {iteration:6d}: "
                      f"Δω_max = {total_change_metric:.2e}, "
                      f"Δψ_max = {psi_max_change:.2e}, "
                      f"ψ_min = {self.psi.min():.6f}")

        elapsed_time = time.time() - start_time
        print(f"\n[WARNING] Max iterations ({max_iterations}) reached.")
        print(f"Final Δω_max = {total_change_metric:.2e}, ψ_min = {self.psi.min():.6f}")
        print(f"Time: {elapsed_time:.2f} seconds")
        
        self.calculate_velocities()
        self.calculate_pressure()
        return False, max_iterations
    
    def calculate_pressure(self):
        """Calculate pressure field using Jacobi Iteration"""
        N = self.N
        h = self.h
        
        rhs = np.zeros((N, N))
        
        # Calculate RHS for pressure Poisson equation: ∇²p = -∇.(u.∇u)
        # for i in range(1, N-1):
        #     for j in range(1, N-1):
        #         dudx = (self.u[i, j+1] - self.u[i, j-1]) / (2*h)
        #         dudy = (self.u[i+1, j] - self.u[i-1, j]) / (2*h)
        #         dvdx = (self.v[i, j+1] - self.v[i, j-1]) / (2*h)
        #         dvdy = (self.v[i+1, j] - self.v[i-1, j]) / (2*h)
                
        #         rhs[i, j] = -(dudx**2 + 2*dudy*dvdx + dvdy**2)
        dudx = (self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * h)
        dudy = (self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * h)
        dvdx = (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * h)
        dvdy = (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * h)
        rhs[1:-1, 1:-1] = -(dudx**2 + 2.0 * dudy * dvdx + dvdy**2)
        
        # Jacobi Iteration for Pressure
        p_new = self.p.copy()
        p_tol = 1e-6
        p_max_iter = 10000 
        
        for p_iter in range(p_max_iter):
            p_old = p_new.copy()
            # Jacobi update
            p_new[1:-1, 1:-1] = 0.25 * (
                    p_old[2:, 1:-1] + p_old[0:-2, 1:-1] + 
                    p_old[1:-1, 2:] + p_old[1:-1, 0:-2] - 
                    h**2 * rhs[1:-1, 1:-1]
                )

            # Neumann BC approximation (dp/dn = 0) - implemented by not updating boundaries, 
            # and normalizing the interior
            
            p_change = np.max(np.abs(p_new - p_old))
            if p_change < p_tol:
                # print(f"  Pressure solver converged in {p_iter} iterations.")
                break
        
        self.p = p_new
        # Normalize pressure
        self.p = self.p - np.mean(self.p[1:-1, 1:-1])

    def create_plots(self, save_fig=True):
        """Create plots with physically consistent coordinate orientation"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure
        fig = plt.figure(figsize=(16, 10), dpi=325)
        
        # 1. Streamlines with CORRECT orientation
        ax1 = plt.subplot(231)
        levels = np.linspace(self.psi.min(), 0, 21)
        
        # Plot with correct x,y orientation
        ax1.contourf(
            self.X, self.Y, self.psi,
            levels=levels,
            cmap='viridis',
            alpha=0.6,
            zorder=1
        )

        ax1.contour(
            self.X, self.Y, self.psi,
            levels=levels,
            colors='#2E86AB',
            linewidths=0.8,
            zorder=2
        )

        ax1.set_xlabel('x (horizontal)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('y (vertical)', fontsize=11, fontweight='bold')
        ax1.set_title('Streamlines ψ', fontsize=12, fontweight='bold')
        ax1.grid(False)
        ax1.set_aspect('equal')
        ax1.set_xlim(0, self.L)
        ax1.set_ylim(0, self.L)
        
        # 2. Vorticity
        ax2 = plt.subplot(232)
        contour2 = ax2.contourf(self.X, self.Y, self.omega, levels=40, 
                               cmap='coolwarm', alpha=0.9)
        ax2.set_xlabel('x (horizontal)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('y (vertical)', fontsize=11, fontweight='bold')
        ax2.set_title('Vorticity ω', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')
        plt.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Velocity magnitude
        ax3 = plt.subplot(233)
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        contour3 = ax3.contourf(self.X, self.Y, vel_mag, levels=30, 
                               cmap='plasma', alpha=0.9)
        ax3.set_xlabel('x (horizontal)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('y (vertical)', fontsize=11, fontweight='bold')
        ax3.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_aspect('equal')
        plt.colorbar(contour3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. u-velocity at vertical centerline (x = L/2)
        ax4 = plt.subplot(234)
        x_idx = np.argmin(np.abs(self.x - self.L/2))
        u_centerline = self.u[:, x_idx]  # All y at fixed x
        
        ax4.plot(u_centerline, self.y, 'b-', linewidth=2.5, 
                label=f'u at x = {self.L/2:.2f}', marker='o', markersize=3)

        ax4.set_xlabel('u-velocity (horizontal)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('y (vertical)', fontsize=11, fontweight='bold')
        ax4.set_title('u-Velocity Profile\nat Vertical Centerline', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(fontsize=9, framealpha=0.9)
        
        # 5. v-velocity at horizontal centerline (y = L/2)
        ax5 = plt.subplot(235)
        y_idx = np.argmin(np.abs(self.y - self.L/2))
        v_centerline = self.v[y_idx, :]  # All x at fixed y
        
        ax5.plot(self.x, v_centerline, 'r-', linewidth=2.5, 
                label=f'v at y = {self.L/2:.2f}', marker='s', markersize=3)
        
        ax5.set_xlabel('x (horizontal)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('v-velocity (vertical)', fontsize=11, fontweight='bold')
        ax5.set_title('v-Velocity Profile\nat Horizontal Centerline',
                     fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.legend(fontsize=9, framealpha=0.9)
        
        # 6. Pressure field
        ax6 = plt.subplot(236)
        contour6 = ax6.contourf(self.X, self.Y, self.p, levels=30, 
                               cmap='RdGy', alpha=0.9)
        ax6.set_xlabel('x (horizontal)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('y (vertical)', fontsize=11, fontweight='bold')
        ax6.set_title('Pressure Field', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.set_aspect('equal')
        plt.colorbar(contour6, ax=ax6, fraction=0.046, pad=0.04)
        
        # Main title
        plt.suptitle(f'Lid-Driven Cavity Flow | Re = {self.Re} | Grid = {self.N}×{self.N}\n'
                    f'Moving lid at top (y = {self.L})', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_fig:
            filename = f'lid_driven_correct_coords_Re{self.Re}_N{self.N}_325dpi.png'
            plt.savefig(filename, dpi=325, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[SUCCESS] Plot saved: {filename}")
        
        plt.show()
        
        # Convergence plot
        if self.history['iterations']:
            fig2, ax = plt.subplots(figsize=(10, 6), dpi=325)
            iterations = self.history['iterations']
            ax.semilogy(iterations, self.history['max_change'], 
                       'b-', linewidth=2.5, label='max Δω')
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Maximum Change', fontsize=11, fontweight='bold')
            ax.set_title('Convergence History', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, framealpha=0.9)
            
            if save_fig:
                conv_filename = f'convergence_Re{self.Re}_N{self.N}_325dpi.png'
                plt.savefig(conv_filename, dpi=325, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                print(f"[SUCCESS] Convergence plot saved: {conv_filename}")
            
            plt.show()
    
    def get_vortex_center(self):
        """
        Get vortex center with CORRECT physical coordinates
        Returns: (x, y) where:
        - x: horizontal position (0 to L)
        - y: vertical position (0 to L)
        """
        # Find minimum streamfunction (primary vortex)
        min_idx = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        # min_idx = (y_index, x_index)
        
        # Convert to physical coordinates
        vortex_x = self.x[min_idx[1]]  # x-coordinate from x-index
        vortex_y = self.y[min_idx[0]]  # y-coordinate from y-index
        
        return vortex_x, vortex_y
    
    def print_summary(self):
        """Print summary with CORRECT coordinates"""
        vortex_x, vortex_y = self.get_vortex_center()
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        
        print(f"\n{'='*60}")
        print("SOLUTION SUMMARY (CORRECT COORDINATES)")
        print(f"{'='*60}")
        print(f"Grid: {self.N} × {self.N}")
        print(f"Reynolds: Re = {self.Re}")
        print(f"Lid velocity: U = {self.U} (at top wall, y = {self.L})")
        print(f"\nFlow Characteristics:")
        print(f"  Streamfunction ψ: min = {self.psi.min():.6f}")
        print(f"  Vorticity ω: min = {self.omega.min():.3f}, max = {self.omega.max():.3f}")
        print(f"  Max velocity: {vel_mag.max():.4f}")
        print(f"  Pressure: min = {self.p.min():.3f}, max = {self.p.max():.3f}")
        print(f"\nVortex Center:")
        print(f"  x = {vortex_x:.4f} (horizontal from left)")
        print(f"  y = {vortex_y:.4f} (vertical from bottom)")
        print(f"{'='*60}")
    
    def save_model(self, filename=None):
        """Save model with CORRECT coordinate info"""
        if filename is None:
            filename = f'lid_driven_model_Re{self.Re}_N{self.N}.pkl'
        
        vortex_x, vortex_y = self.get_vortex_center()
        
        model_data = {
            'parameters': {
                'N': self.N,
                'Re': self.Re,
                'U': self.U,
                'L': self.L,
                'h': self.h,
                'nu': self.nu,
                'dt': self.dt
            },
            'coordinates': {
                'x': self.x,  # Horizontal
                'y': self.y,  # Vertical
                'X': self.X,  # X[i,j] = x-coordinate at (i,j)
                'Y': self.Y,  # Y[i,j] = y-coordinate at (i,j)
                'note': 'X,Y are physical coordinates: X=horizontal, Y=vertical'
            },
            'fields': {
                'psi': self.psi,    # ψ[i,j] at (x[j], y[i])
                'omega': self.omega,
                'u': self.u,        # u[i,j] = horizontal velocity at (x[j], y[i])
                'v': self.v,        # v[i,j] = vertical velocity at (x[j], y[i])
                'p': self.p
            },
            'vortex_center': {
                'x': vortex_x,
                'y': vortex_y,
                'array_indices': np.unravel_index(np.argmin(self.psi), self.psi.shape)
            },
            'convergence': self.history,
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'coordinate_system': 'x=horizontal (0 to L), y=vertical (0 to L)',
                'moving_lid': f'top wall (y = {self.L})',
                'note': 'Array indexing: [y_index, x_index]'
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[SUCCESS] Model saved: {filename}")
        
        # Save numpy arrays
        np.savez(f'flow_fields_Re{self.Re}_N{self.N}.npz',
                x=self.x, y=self.y,
                X=self.X, Y=self.Y,
                psi=self.psi, omega=self.omega,
                u=self.u, v=self.v, p=self.p,
                vortex_x=vortex_x, vortex_y=vortex_y)
        print(f"[SUCCESS] Numpy arrays saved")

    def create_showcase_plots(self, save_fig=True):
        plt.style.use('seaborn-v0_8-white')

        vortex_x, vortex_y = self.get_vortex_center()

        fig, axs = plt.subplots(1, 2, figsize=(22, 10), dpi=300)

        TITLE_FS = 16
        LABEL_FS = 13
        TICK_FS  = 11

        # ==========================================================
        # 1. STREAMLINES (ψ) — HERO PLOT
        # ==========================================================
        ax = axs[0]

        levels = np.linspace(self.psi.min(), 0, 25)

        ax.contourf(
            self.X, self.Y, self.psi,
            levels=levels,
            cmap='viridis',
            alpha=0.85,
            zorder=1
        )

        ax.contour(
            self.X, self.Y, self.psi,
            levels=levels,
            colors='black',
            linewidths=1.3,
            alpha=0.6,
            zorder=2
        )

        ax.set_title("Streamlines (ψ)", fontsize=TITLE_FS, fontweight='bold')
        ax.set_xlabel("x", fontsize=LABEL_FS)
        ax.set_ylabel("y", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)

        ax.set_aspect('equal')
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.grid(False)

        # ==========================================================
        # 2. CENTERLINE U-VELOCITY — VALIDATION PLOT
        # ==========================================================
        ax = axs[1]

        x_idx = np.argmin(np.abs(self.x - self.L / 2))
        u_centerline = self.u[:, x_idx]

        ax.plot(
            u_centerline,
            self.y,
            color='black',
            linewidth=3.2,
            label=r"$u(y)$ at $x = L/2$"
        )

        ax.set_title("Centerline Velocity Profile", fontsize=TITLE_FS, fontweight='bold')
        ax.set_xlabel("u-velocity", fontsize=LABEL_FS)
        ax.set_ylabel("y", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, framealpha=0.95)

        # ==========================================================
        # SUPERTITLE
        # ==========================================================
        plt.suptitle(
            f"Lid-Driven Cavity Flow — Re = {self.Re}, Grid = {self.N}×{self.N}",
            fontsize=18,
            fontweight='bold',
            y=0.98
        )

        plt.subplots_adjust(wspace=0.25)

        if save_fig:
            fname = f"lid_driven_showcase_centerline_Re{self.Re}_N{self.N}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Showcase plot saved: {fname}")

        plt.show()

# ============================================================================
# EXECUTION BLOCK
# ============================================================================
if __name__ == '__main__':
    # Example configuration (computationally expensive)
    N_GRID = 256        # Fine Grid size
    REYNOLDS = 50     # High Reynolds number
    # WARNING: High Reynolds number and small time step lead to a large number of iterations.
    # Recommended for testing: N=101, Re=100
    MAX_ITER = 100000   
    TOLERANCE = 1e-5    # Target convergence tolerance

    # --- Run the High Re Simulation ---
    solver = LidDrivenCavitySolver(N=N_GRID, Re=REYNOLDS)
    converged, iterations = solver.solve(max_iterations=MAX_ITER, tolerance=TOLERANCE)

    # --- Post-Processing ---
    if converged or iterations == MAX_ITER:
        solver.print_summary()
        solver.create_plots(save_fig=False)
        solver.create_showcase_plots(save_fig=False)
        solver.save_model()
