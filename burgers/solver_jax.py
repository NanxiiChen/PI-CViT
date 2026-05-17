import os
import jax
import jax.numpy as jnp
from jax import lax


def _build_wavenumbers(Nx, Ny, Lx, Ly, dtype=jnp.float32):
    kx = (2.0 * jnp.pi) * jnp.fft.fftfreq(Nx, d=Lx / Nx).astype(dtype)
    ky = (2.0 * jnp.pi) * jnp.fft.fftfreq(Ny, d=Ly / Ny).astype(dtype)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def _build_dealias_filter(KX, KY, Nx, Ny, Lx, Ly):
    """2/3 规则去混叠滤波器，返回 float32 掩码（1=保留，0=截断）。"""
    kx_max = (2.0 * jnp.pi) * (Nx // 2) / Lx
    ky_max = (2.0 * jnp.pi) * (Ny // 2) / Ly
    mask = (jnp.abs(KX) <= (2.0 / 3.0) * kx_max) & (jnp.abs(KY) <= (2.0 / 3.0) * ky_max)
    return mask.astype(jnp.float32)


def compute_rhs(u, v, KX, KY, laplacian, dealias, nu):
    """
    计算 2D 矢量 Burgers 方程右端项。

    方程：
        u_t = - u*u_x - v*u_y + nu*(u_xx + u_yy)
        v_t = - u*v_x - v*v_y + nu*(v_xx + v_yy)
    """
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)

    # 空间导数（谱方法）
    u_x = jnp.fft.ifft2(1j * KX * u_hat).real
    u_y = jnp.fft.ifft2(1j * KY * u_hat).real
    v_x = jnp.fft.ifft2(1j * KX * v_hat).real
    v_y = jnp.fft.ifft2(1j * KY * v_hat).real

    # 非线性项（物理空间相乘后回频域并去混叠）
    nl_u_hat = jnp.fft.fft2(-(u * u_x + v * u_y)) * dealias
    nl_v_hat = jnp.fft.fft2(-(u * v_x + v * v_y)) * dealias

    # 黏性项（频域直接相乘）
    visc_u_hat = nu * laplacian * u_hat
    visc_v_hat = nu * laplacian * v_hat

    rhs_u = jnp.fft.ifft2(nl_u_hat + visc_u_hat).real
    rhs_v = jnp.fft.ifft2(nl_v_hat + visc_v_hat).real

    return rhs_u, rhs_v


def rk4_step(u, v, dt, KX, KY, laplacian, dealias, nu):
    k1_u, k1_v = compute_rhs(u, v, KX, KY, laplacian, dealias, nu)

    k2_u, k2_v = compute_rhs(
        u + 0.5 * dt * k1_u,
        v + 0.5 * dt * k1_v,
        KX, KY, laplacian, dealias, nu,
    )

    k3_u, k3_v = compute_rhs(
        u + 0.5 * dt * k2_u,
        v + 0.5 * dt * k2_v,
        KX, KY, laplacian, dealias, nu,
    )

    k4_u, k4_v = compute_rhs(
        u + dt * k3_u,
        v + dt * k3_v,
        KX, KY, laplacian, dealias, nu,
    )

    u_new = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return u_new, v_new


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


def _solve_one_ic_core(u0, v0, KX, KY, laplacian, dealias, nu, dt, n_steps, save_every, n_saves):
    Nx, Ny = u0.shape

    u_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=u0.dtype).at[0].set(u0)
    v_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=v0.dtype).at[0].set(v0)

    def body(carry, step):
        u_now, v_now, save_idx, u_hist, v_hist = carry
        u_new, v_new = rk4_step(u_now, v_now, dt, KX, KY, laplacian, dealias, nu)
        should_save = (step % save_every) == 0

        def do_save(vals):
            u_s, v_s, idx_s, uh_s, vh_s = vals
            uh_s = uh_s.at[idx_s].set(u_s)
            vh_s = vh_s.at[idx_s].set(v_s)
            return u_s, v_s, idx_s + 1, uh_s, vh_s

        u_new, v_new, save_idx, u_hist, v_hist = lax.cond(
            should_save,
            do_save,
            lambda vals: vals,
            (u_new, v_new, save_idx, u_hist, v_hist),
        )

        return (u_new, v_new, save_idx, u_hist, v_hist), None

    init_carry = (u0, v0, 1, u_hist0, v_hist0)
    steps = jnp.arange(1, n_steps + 1)
    final_carry, _ = lax.scan(body, init_carry, xs=steps)
    _, _, _, u_hist, v_hist = final_carry
    return u_hist, v_hist


def solve_multiple_ics(initial_conditions, Nx, Ny, nu, t_end, dt, save_interval, Lx=1.0, Ly=1.0):
    """
    对多个初始条件并行求解（JAX vmap + jit）。

    initial_conditions: (N, 2, Nx, Ny) — 通道 0=u，通道 1=v
    返回:
        solutions : (N, T, 2, Nx, Ny)
        save_times: (T,)
    """
    n_steps, save_every, n_saves = _validate_time_grid(t_end, dt, save_interval)
    save_times = jnp.arange(n_saves, dtype=jnp.float32) * (save_every * dt)

    KX, KY = _build_wavenumbers(Nx, Ny, Lx, Ly, dtype=jnp.float32)
    laplacian = -(KX ** 2 + KY ** 2)
    dealias = _build_dealias_filter(KX, KY, Nx, Ny, Lx, Ly)
    nu_jax = jnp.asarray(nu, dtype=jnp.float32)

    ics_jax = jnp.asarray(initial_conditions, dtype=jnp.float32)
    u0_all = ics_jax[:, 0]  # (N, Nx, Ny)
    v0_all = ics_jax[:, 1]  # (N, Nx, Ny)

    one_sample_solver = jax.jit(
        lambda u0, v0: _solve_one_ic_core(
            u0, v0, KX, KY, laplacian, dealias, nu_jax,
            dt, n_steps, save_every, n_saves,
        )
    )

    u_hist_all, v_hist_all = jax.vmap(one_sample_solver, in_axes=(0, 0))(u0_all, v0_all)
    solutions = jnp.stack([u_hist_all, v_hist_all], axis=2)  # (N, T, 2, Nx, Ny)

    return solutions, save_times


def generate_periodic_ic_jax(key, Nx, Ny, length_scale=0.1, amplitude=0.01, Lx=1.0, Ly=1.0):
    """使用 JAX PRNG 生成单个周期性高斯随机场初值。"""
    kx = jnp.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2

    spectrum = jnp.exp(-0.5 * (length_scale ** 2) * K2)
    key_re, key_im = jax.random.split(key)
    noise_re = jax.random.normal(key_re, shape=(Nx, Ny), dtype=jnp.float32)
    noise_im = jax.random.normal(key_im, shape=(Nx, Ny), dtype=jnp.float32)
    u_hat = (noise_re + 1j * noise_im) * jnp.sqrt(spectrum)

    field = jnp.fft.ifft2(u_hat).real
    field = (field - jnp.mean(field)) / (jnp.std(field) + 1e-12)
    return amplitude * field


def generate_periodic_ics_batch_jax(keys, Nx, Ny, length_scale=0.1, amplitude=0.01, Lx=1.0, Ly=1.0):
    sampler = lambda k: generate_periodic_ic_jax(
        k, Nx, Ny,
        length_scale=length_scale,
        amplitude=amplitude,
        Lx=Lx,
        Ly=Ly,
    )
    return jax.vmap(sampler)(keys)


if __name__ == "__main__":
    N_samples = 100
    Nx, Ny = 64, 64
    nu = 0.01
    t_end = 1.0
    dt = 0.001
    save_interval = 0.1

    ic_params = {
        "length_scale": 0.1,
        "amplitude": 0.2,
        "seed": 543,
    }

    print("=" * 60)
    print("2D Vector Burgers Equation Solver (JAX all-jnp pipeline)")
    print(f"Grid: {Nx}x{Ny}, Samples: {N_samples}")
    print(f"nu = {nu}")
    print("=" * 60)

    print("\nGenerating Initial Conditions with JAX PRNG...")
    key = jax.random.PRNGKey(ic_params["seed"])
    # 每个样本的 u、v 分量各需要一个独立的 key
    keys = jax.random.split(key, 2 * N_samples)
    keys_u = keys[:N_samples]
    keys_v = keys[N_samples:]

    u0_all = generate_periodic_ics_batch_jax(
        keys_u, Nx, Ny,
        length_scale=ic_params["length_scale"],
        amplitude=ic_params["amplitude"],
    )
    v0_all = generate_periodic_ics_batch_jax(
        keys_v, Nx, Ny,
        length_scale=ic_params["length_scale"],
        amplitude=ic_params["amplitude"],
    )
    initial_conditions = jnp.stack([u0_all, v0_all], axis=1)  # (N, 2, Nx, Ny)
    print(f"Initial conditions shape: {initial_conditions.shape}")

    print("\nSolving Equations with JAX parallel vmap...")
    solutions, times = solve_multiple_ics(
        initial_conditions,
        Nx,
        Ny,
        nu,
        t_end,
        dt,
        save_interval,
    )

    print(f"\n求解完成！")
    print(f"解的形状: {tuple(solutions.shape)} (N, T, C, Nx, Ny), Channels: 0=u, 1=v")
    print(f"保存的时间点: {times}")

    # 简单诊断：检查总动能随时间的变化（黏性耗散应使能量单调下降）
    ke = 0.5 * jnp.mean(solutions[0, :, 0] ** 2 + solutions[0, :, 1] ** 2, axis=(-1, -2))
    print(f"\nDiagnostics (sample 0): KE[0]={float(ke[0]):.4f}, KE[-1]={float(ke[-1]):.4f}, "
          f"change={float(ke[-1] - ke[0]):.4e}")

    print("\nSaving Results...")
    os.makedirs("./data/burgers", exist_ok=True)
    save_path = "./data/burgers/burgers_solutions.npz"

    x = jnp.linspace(0.0, 1.0, Nx, endpoint=False, dtype=jnp.float32)
    y = jnp.linspace(0.0, 1.0, Ny, endpoint=False, dtype=jnp.float32)

    jnp.savez(
        save_path,
        solutions=jnp.transpose(solutions, (0, 1, 2, 4, 3)),  # (N, T, C, Ny, Nx)
        times=times,
        x=x,
        y=y,
        nu=nu,
        dt=dt,
    )
    print(f"Saved to {save_path}")

    try:
        import matplotlib.pyplot as plt

        os.makedirs("tmp", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        plot_indices = [0, 2, 4, 6, 8, 10]

        for i, t_idx in enumerate(plot_indices):
            u_field = solutions[0, t_idx, 0, :, :]
            im = axes[i].contourf(x, y, u_field.T, levels=20, cmap="RdBu_r")
            axes[i].set_title(f"u (t={float(times[t_idx]):.2f})")
            plt.colorbar(im, ax=axes[i])
            axes[i].set_aspect("equal")

        plt.tight_layout()
        plt.savefig("tmp/burgers_evolution_u_jax.png")
        print("Visualization saved to tmp/burgers_evolution_u_jax.png")
    except ImportError:
        print("Matplotlib not found, skipping visualization.")

