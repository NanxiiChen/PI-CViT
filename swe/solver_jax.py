import os
import jax
import jax.numpy as jnp
from jax import lax


def _build_wavenumbers(Nx, Ny, Lx, Ly, dtype=jnp.float32):
    kx = (2.0 * jnp.pi) * jnp.fft.fftfreq(Nx, d=Lx / Nx).astype(dtype)
    ky = (2.0 * jnp.pi) * jnp.fft.fftfreq(Ny, d=Ly / Ny).astype(dtype)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def compute_rhs(h, u, v, KX, KY, H, f, g):
    h_hat = jnp.fft.fft2(h)
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)

    h_x_hat = 1j * KX * h_hat
    h_y_hat = 1j * KY * h_hat
    u_x_hat = 1j * KX * u_hat
    v_y_hat = 1j * KY * v_hat

    rhs_h_hat = -H * (u_x_hat + v_y_hat)
    rhs_u_hat = f * v_hat - g * h_x_hat
    rhs_v_hat = -f * u_hat - g * h_y_hat

    rhs_h = jnp.fft.ifft2(rhs_h_hat).real
    rhs_u = jnp.fft.ifft2(rhs_u_hat).real
    rhs_v = jnp.fft.ifft2(rhs_v_hat).real

    return rhs_h, rhs_u, rhs_v


def rk4_step(h, u, v, dt, KX, KY, H, f, g):
    k1_h, k1_u, k1_v = compute_rhs(h, u, v, KX, KY, H, f, g)

    k2_h, k2_u, k2_v = compute_rhs(
        h + 0.5 * dt * k1_h,
        u + 0.5 * dt * k1_u,
        v + 0.5 * dt * k1_v,
        KX,
        KY,
        H,
        f,
        g,
    )

    k3_h, k3_u, k3_v = compute_rhs(
        h + 0.5 * dt * k2_h,
        u + 0.5 * dt * k2_u,
        v + 0.5 * dt * k2_v,
        KX,
        KY,
        H,
        f,
        g,
    )

    k4_h, k4_u, k4_v = compute_rhs(
        h + dt * k3_h,
        u + dt * k3_u,
        v + dt * k3_v,
        KX,
        KY,
        H,
        f,
        g,
    )

    h_new = h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
    u_new = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return h_new, u_new, v_new


def _validate_time_grid(t_end, dt, save_interval):
    n_steps_float = t_end / dt
    n_steps = int(round(n_steps_float))
    if abs(n_steps - n_steps_float) > 1e-10:
        raise ValueError(f"t_end/dt must be an integer, got {n_steps_float}")

    save_every_float = save_interval / dt
    save_every = int(round(save_every_float))
    if abs(save_every - save_every_float) > 1e-10:
        raise ValueError(f"save_interval/dt must be an integer, got {save_every_float}")

    if save_every <= 0:
        raise ValueError("save_every must be positive")

    if n_steps % save_every != 0:
        raise ValueError("n_steps must be divisible by save_every to keep exact save grid")

    n_saves = n_steps // save_every + 1
    return n_steps, save_every, n_saves


def _solve_one_ic_core(h0, KX, KY, H, f, g, dt, n_steps, save_every, n_saves):
    Nx, Ny = h0.shape
    u0 = jnp.zeros((Nx, Ny), dtype=h0.dtype)
    v0 = jnp.zeros((Nx, Ny), dtype=h0.dtype)

    h_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=h0.dtype).at[0].set(h0)
    u_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=h0.dtype).at[0].set(u0)
    v_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=h0.dtype).at[0].set(v0)

    def body(carry, step):
        h_now, u_now, v_now, save_idx, h_hist, u_hist, v_hist = carry
        h_new, u_new, v_new = rk4_step(h_now, u_now, v_now, dt, KX, KY, H, f, g)
        should_save = (step % save_every) == 0

        def do_save(vals):
            h_s, u_s, v_s, idx_s, hh_s, uh_s, vh_s = vals
            hh_s = hh_s.at[idx_s].set(h_s)
            uh_s = uh_s.at[idx_s].set(u_s)
            vh_s = vh_s.at[idx_s].set(v_s)
            return h_s, u_s, v_s, idx_s + 1, hh_s, uh_s, vh_s

        h_new, u_new, v_new, save_idx, h_hist, u_hist, v_hist = lax.cond(
            should_save,
            do_save,
            lambda vals: vals,
            (h_new, u_new, v_new, save_idx, h_hist, u_hist, v_hist),
        )

        return (h_new, u_new, v_new, save_idx, h_hist, u_hist, v_hist), None

    init_carry = (h0, u0, v0, 1, h_hist0, u_hist0, v_hist0)
    steps = jnp.arange(1, n_steps + 1)
    final_carry, _ = lax.scan(body, init_carry, xs=steps)
    _, _, _, _, h_hist, u_hist, v_hist = final_carry
    return h_hist, u_hist, v_hist


def solve_multiple_ics(initial_h, Nx, Ny, params, t_end, dt, save_interval):
    """
    对多个初始条件并行求解（JAX vmap + jit）。

    initial_h: (N, Nx, Ny)
    返回:
    solutions: (N, T, 3, Nx, Ny)
    save_times: (T,)
    """
    H = jnp.asarray(params["H"], dtype=jnp.float32)
    f = jnp.asarray(params["f"], dtype=jnp.float32)
    g = jnp.asarray(params["g"], dtype=jnp.float32)
    Lx = jnp.asarray(params["Lx"], dtype=jnp.float32)
    Ly = jnp.asarray(params["Ly"], dtype=jnp.float32)

    n_steps, save_every, n_saves = _validate_time_grid(t_end, dt, save_interval)
    save_times = jnp.arange(n_saves, dtype=jnp.float32) * (save_every * dt)

    KX, KY = _build_wavenumbers(Nx, Ny, Lx, Ly, dtype=jnp.float32)
    initial_h_jax = jnp.asarray(initial_h, dtype=jnp.float32)

    one_sample_solver = jax.jit(
        lambda h0: _solve_one_ic_core(
            h0,
            KX,
            KY,
            H,
            f,
            g,
            dt,
            n_steps,
            save_every,
            n_saves,
        )
    )

    h_hist_all, u_hist_all, v_hist_all = jax.vmap(one_sample_solver, in_axes=0)(initial_h_jax)
    solutions = jnp.stack([h_hist_all, u_hist_all, v_hist_all], axis=2)

    return solutions, save_times


def generate_periodic_ics_h_jax(key, Nx, Ny, length_scale=0.1, amplitude=1.0, Lx=1.0, Ly=1.0):
    """使用 JAX PRNG 生成单个周期性高斯随机场初值。"""
    kx = jnp.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    spectrum = jnp.exp(-0.5 * (length_scale**2) * K2)
    key_re, key_im = jax.random.split(key)
    noise_re = jax.random.normal(key_re, shape=(Nx, Ny), dtype=jnp.float32)
    noise_im = jax.random.normal(key_im, shape=(Nx, Ny), dtype=jnp.float32)
    u_hat = (noise_re + 1j * noise_im) * jnp.sqrt(spectrum)

    field = jnp.fft.ifft2(u_hat).real
    field = (field - jnp.mean(field)) / (jnp.std(field) + 1e-12)
    return amplitude * field


def generate_periodic_ics_batch_jax(keys, Nx, Ny, length_scale=0.1, amplitude=1.0, Lx=1.0, Ly=1.0):
    sampler = lambda k: generate_periodic_ics_h_jax(
        k,
        Nx,
        Ny,
        length_scale=length_scale,
        amplitude=amplitude,
        Lx=Lx,
        Ly=Ly,
    )
    return jax.vmap(sampler)(keys)


if __name__ == "__main__":
    N_samples = 1024
    Nx, Ny = 64, 64
    t_end = 1.0
    dt = 0.001
    save_interval = 0.01

    swe_params = {
        "Lx": 1.0,
        "Ly": 1.0,
        "H": 1.0,
        "f": 10.0,
        "g": 1.0,
    }

    ic_params = {
        "length_scale": 0.1,
        "amplitude": 0.5,
        "seed": 0,
    }

    print("=" * 60)
    print("Linear Shallow Water Equation Solver (JAX all-jnp pipeline)")
    print(f"Grid: {Nx}x{Ny}, Samples: {N_samples}")
    print(f"Physics: {swe_params}")
    print("=" * 60)

    print("\nGenerating Initial Conditions with JAX PRNG...")
    key = jax.random.PRNGKey(ic_params["seed"])
    keys = jax.random.split(key, N_samples)
    initial_h = generate_periodic_ics_batch_jax(
        keys,
        Nx,
        Ny,
        length_scale=ic_params["length_scale"],
        amplitude=ic_params["amplitude"],
        Lx=swe_params["Lx"],
        Ly=swe_params["Ly"],
    )

    print("\nSolving Equations with JAX parallel vmap...")
    solutions, times = solve_multiple_ics(
        initial_h,
        Nx,
        Ny,
        swe_params,
        t_end,
        dt,
        save_interval,
    )

    h_hist = solutions[0, :, 0, :, :]
    u_hist = solutions[0, :, 1, :, :]
    v_hist = solutions[0, :, 2, :, :]

    mean_h = jnp.mean(h_hist, axis=(1, 2))
    energy = 0.5 * (
        swe_params["g"] * jnp.mean(h_hist**2, axis=(1, 2))
        + swe_params["H"] * jnp.mean(u_hist**2 + v_hist**2, axis=(1, 2))
    )

    abs_mean_change = jnp.max(mean_h) - jnp.min(mean_h)
    print(f"Diagnostics (sample 0): mean(h) abs change = {float(abs_mean_change):.3e}")

    h_hat_all = jnp.fft.fft2(h_hist, axes=(1, 2))
    h0_mode = h_hat_all[:, 0, 0]
    h0_abs_change = jnp.max(jnp.abs(h0_mode)) - jnp.min(jnp.abs(h0_mode))
    h0_mean_abs = jnp.mean(jnp.abs(h0_mode))
    h0_rel_change = jnp.where(h0_mean_abs < 1e-12, jnp.nan, h0_abs_change / h0_mean_abs)
    rel_txt = "nan" if bool(jnp.isnan(h0_rel_change)) else f"{float(h0_rel_change):.3e}"
    print(f"k=0 mode: abs change = {float(h0_abs_change):.3e}, rel change = {rel_txt}")

    rel_energy_change = (jnp.max(energy) - jnp.min(energy)) / (jnp.mean(jnp.abs(energy)) + 1e-16)
    print(f"Diagnostics (sample 0): energy change rel = {float(rel_energy_change):.3e}")

    print("\nSaving Results...")
    os.makedirs("./data/swe", exist_ok=True)
    save_path = "./data/swe/f10/swe_training.npz"

    x = jnp.linspace(0.0, 1.0, Nx, endpoint=False, dtype=jnp.float32)
    y = jnp.linspace(0.0, 1.0, Ny, endpoint=False, dtype=jnp.float32)

    jnp.savez(
        save_path,
        solutions=jnp.transpose(solutions, (0, 1, 2, 4, 3)),
        times=times,
        x=x,
        y=y,
        **swe_params,
    )
    print(f"Saved to {save_path}")
    print(f"Shape: {tuple(solutions.shape)} (N, T, C, Nx, Ny), Channels: 0=h, 1=u, 2=v")

    try:
        import matplotlib.pyplot as plt

        os.makedirs("tmp", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        plot_indices = [0, 2, 4, 6, 8, 10]

        for i, t_idx in enumerate(plot_indices):
            h_field = solutions[0, t_idx, 0, :, :]
            im = axes[i].contourf(x, y, h_field.T, levels=20, cmap="RdBu_r")
            axes[i].set_title(f"h (t={float(times[t_idx]):.2f})")
            plt.colorbar(im, ax=axes[i])
            axes[i].set_aspect("equal")

        plt.tight_layout()
        plt.savefig("tmp/swe_evolution_h_jax.png")
        print("Visualization saved to tmp/swe_evolution_h_jax.png")
    except ImportError:
        print("Matplotlib not found, skipping visualization.")
