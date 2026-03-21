
import os
import jax
import jax.numpy as jnp
from jax import lax


def _build_spectral_ops(Nx, Ny, Lx=1.0, Ly=1.0, dtype=jnp.float32):
    kx = (2.0 * jnp.pi) * jnp.fft.fftfreq(Nx, d=jnp.asarray(Lx, dtype=dtype) / Nx).astype(dtype)
    ky = (2.0 * jnp.pi) * jnp.fft.fftfreq(Ny, d=jnp.asarray(Ly, dtype=dtype) / Ny).astype(dtype)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    lap = -(KX**2 + KY**2)

    # 2/3 去混叠掩码
    kx_cut = (2.0 / 3.0) * jnp.max(jnp.abs(kx))
    ky_cut = (2.0 / 3.0) * jnp.max(jnp.abs(ky))
    dealias = (jnp.abs(KX) <= kx_cut) & (jnp.abs(KY) <= ky_cut)
    return lap, dealias


def _spectral_laplacian(u, lap, dealias):
    u_hat = jnp.fft.fft2(u)
    lap_u_hat = lap * u_hat
    lap_u_hat = jnp.where(dealias, lap_u_hat, 0.0 + 0.0j)
    return jnp.fft.ifft2(lap_u_hat).real


def compute_rhs(u, v, c2_field, lap, dealias):
    rhs_u = v
    lap_u = _spectral_laplacian(u, lap, dealias)
    rhs_v = c2_field * lap_u
    return rhs_u, rhs_v


def rk4_step(u, v, dt, c2_field, lap, dealias):
    k1_u, k1_v = compute_rhs(u, v, c2_field, lap, dealias)
    k2_u, k2_v = compute_rhs(u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v, c2_field, lap, dealias)
    k3_u, k3_v = compute_rhs(u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v, c2_field, lap, dealias)
    k4_u, k4_v = compute_rhs(u + dt * k3_u, v + dt * k3_v, c2_field, lap, dealias)

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
        raise ValueError("n_steps must be divisible by save_every")

    n_saves = n_steps // save_every + 1
    return n_steps, save_every, n_saves


def _solve_one_ic_core(u0, c2_field, dt, n_steps, save_every, n_saves, lap, dealias):
    Nx, Ny = u0.shape
    v0 = jnp.zeros((Nx, Ny), dtype=u0.dtype)

    u_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=u0.dtype).at[0].set(u0)
    v_hist0 = jnp.zeros((n_saves, Nx, Ny), dtype=u0.dtype).at[0].set(v0)

    def body(carry, step):
        u_now, v_now, save_idx, u_hist, v_hist = carry
        u_new, v_new = rk4_step(u_now, v_now, dt, c2_field, lap, dealias)
        should_save = (step % save_every) == 0

        def do_save(vals):
            uu, vv, idx, uh, vh = vals
            uh = uh.at[idx].set(uu)
            vh = vh.at[idx].set(vv)
            return uu, vv, idx + 1, uh, vh

        u_new, v_new, save_idx, u_hist, v_hist = lax.cond(
            should_save,
            do_save,
            lambda vals: vals,
            (u_new, v_new, save_idx, u_hist, v_hist),
        )
        return (u_new, v_new, save_idx, u_hist, v_hist), None

    init = (u0, v0, 1, u_hist0, v_hist0)
    steps = jnp.arange(1, n_steps + 1, dtype=jnp.int32)
    final, _ = lax.scan(body, init, xs=steps)
    _, _, _, u_hist, v_hist = final
    return u_hist, v_hist


def solve_multiple_ics(initial_conditions, c_fields, Nx, Ny, t_end, dt, save_interval, Lx=1.0, Ly=1.0):
    """
    initial_conditions: (N, Nx, Ny)
    c_fields: (N, Nx, Ny)
    return:
      solutions: (N, T, 1, Nx, Ny)   # 只保存位移 u
      times: (T,)
    """
    dt = jnp.float32(dt)
    t_end = jnp.float32(t_end)
    save_interval = jnp.float32(save_interval)

    n_steps, save_every, n_saves = _validate_time_grid(t_end, dt, save_interval)
    times = jnp.arange(n_saves, dtype=jnp.float32) * (save_every * dt)

    lap, dealias = _build_spectral_ops(Nx, Ny, Lx=Lx, Ly=Ly, dtype=jnp.float32)

    u0_all = jnp.asarray(initial_conditions, dtype=jnp.float32)
    c_all = jnp.asarray(c_fields, dtype=jnp.float32)
    c2_all = c_all**2

    one = jax.jit(
        lambda u0, c2: _solve_one_ic_core(
            u0=u0,
            c2_field=c2,
            dt=dt,
            n_steps=n_steps,
            save_every=save_every,
            n_saves=n_saves,
            lap=lap,
            dealias=dealias,
        )
    )

    u_hist_all, v_hist_all = jax.vmap(one, in_axes=(0, 0))(u0_all, c2_all)

    # 按你原始接口返回位移通道 C=1
    solutions = u_hist_all[:, :, None, :, :]
    return solutions, times


def generate_periodic_field_jax(key, Nx, Ny, length_scale=0.1, amplitude=1.0, Lx=1.0, Ly=1.0):
    kx = jnp.fft.fftfreq(Nx, d=jnp.float32(Lx) / Nx) * (2.0 * jnp.pi)
    ky = jnp.fft.fftfreq(Ny, d=jnp.float32(Ly) / Ny) * (2.0 * jnp.pi)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    spectrum = jnp.exp(-0.5 * (jnp.float32(length_scale) ** 2) * K2)
    k1, k2 = jax.random.split(key)
    noise_re = jax.random.normal(k1, (Nx, Ny), dtype=jnp.float32)
    noise_im = jax.random.normal(k2, (Nx, Ny), dtype=jnp.float32)
    field_hat = (noise_re + 1j * noise_im) * jnp.sqrt(spectrum)

    field = jnp.fft.ifft2(field_hat).real
    field = (field - jnp.mean(field)) / (jnp.std(field) + 1e-12)
    return jnp.float32(amplitude) * field


def generate_periodic_field_batch_jax(keys, Nx, Ny, length_scale=0.1, amplitude=1.0, Lx=1.0, Ly=1.0):
    f = lambda k: generate_periodic_field_jax(
        k, Nx, Ny, length_scale=length_scale, amplitude=amplitude, Lx=Lx, Ly=Ly
    )
    return jax.vmap(f)(keys)


if __name__ == "__main__":
    N = 1024
    Nx, Ny = 64, 64
    t_end = 1.0
    dt = 1e-4
    save_interval = 0.01
    u0_length_scale = 0.1
    u0_amplitude = 0.2
    seed = 123

    print("=" * 60)
    print("2D Wave Solver JAX (u_tt -> first-order system)")
    print("=" * 60)

    # 初始位移场（JAX PRNG）
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, N)
    initial_conditions = generate_periodic_field_batch_jax(
        keys, Nx, Ny, length_scale=u0_length_scale, amplitude=u0_amplitude
    )

    # 波速场（可按样本变化；这里沿用你原来正弦场）
    x = jnp.linspace(0.0, 1.0, Nx, endpoint=False, dtype=jnp.float32)
    y = jnp.linspace(0.0, 1.0, Ny, endpoint=False, dtype=jnp.float32)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    c_single = 1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * X) * jnp.sin(2.0 * jnp.pi * Y)
    c_fields = jnp.repeat(c_single[None, :, :], N, axis=0)

    solutions, times = solve_multiple_ics(
        initial_conditions=initial_conditions,
        c_fields=c_fields,
        Nx=Nx,
        Ny=Ny,
        t_end=t_end,
        dt=dt,
        save_interval=save_interval,
    )

    os.makedirs("./data/wave", exist_ok=True)
    jnp.savez(
        "./data/wave/wave_training.npz",
        solutions=jnp.transpose(solutions, (0, 1, 2, 4, 3)),
        times=times,
        x=x,
        y=y,
        c_fields=jnp.transpose(c_fields, (0, 2, 1)),
        initial_conditions=jnp.transpose(initial_conditions, (0, 2, 1)),
        dt=float(dt),
    )
    print("Saved to ./data/wave/wave_solutions.npz")