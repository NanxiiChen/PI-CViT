from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, repeat

from models.sample import lhs_sampling, shifted_grid

class CoordSampler:
    
    def __init__(
        self,
        spatial_domain: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0, 1.0),
            (0, 1.0)
        ),
        temporal_domain: Tuple[float, float] = (0.0, 1.0),
        num_pde_samples: int = 1024,
    ):
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.num_pde_samples = num_pde_samples
        
        
    def sample_pde(self, key: jax.Array) -> jnp.ndarray:
        """Sample points within the spatio-temporal domain using LHS."""
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, t_max = self.temporal_domain

        mins = jnp.array([x_min, y_min, t_min])
        maxs = jnp.array([x_max, y_max, t_max])
        # maxs_time_refined = jnp.array([x_max, y_max, t_min + 0.1 * (t_max - t_min)])
        # k1, k2 = jax.random.split(key)
        pde_points = lhs_sampling(mins, maxs, self.num_pde_samples, key)
        # pde_points_initial_time_refined = lhs_sampling(
        #     mins,
        #     maxs_time_refined,
        #     self.num_pde_samples // 2,
        #     k2
        # )
        # pde_points = jnp.concatenate([pde_points, pde_points_initial_time_refined], axis=0)

        return pde_points
    
    def resample(
        self,
        key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        pde_samples = self.sample_pde(key)
        
        return {
            "pde": pde_samples
        }
    

class FunctionSampler:
    
    def __init__(
        self,
        lx,
        ly,
        length_scale: float = 0.1,
        amplitude: float = 0.01,
        grid_size=(64, 64),
        num_u_samples: int = 16,
    ):
        self.num_u_samples = num_u_samples
        self.amplitude = amplitude
        self.grid_size = grid_size
        
        Nx, Ny = grid_size
        kx = jnp.fft.fftfreq(Nx, d=lx / Nx) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(Ny, d=ly / Ny) * 2 * jnp.pi
        KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        K2 = K2.at[0, 0].set(1.0)  # avoid division by zero
        self.spectrum = jnp.exp(-0.5 * length_scale**2 * K2)
        
    def sample_one_u(
        self,
        key: jax.Array
    ):
        # noise = np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)
        Nx, Ny = self.grid_size
        key_re, key_im = jax.random.split(key)
        noise_re = jax.random.normal(key_re, shape=(Nx, Ny))
        noise_im = jax.random.normal(key_im, shape=(Nx, Ny))
        noise = noise_re + 1j * noise_im
        u_hat = noise * jnp.sqrt(self.spectrum)
        
        field = jnp.fft.ifft2(u_hat).real
        field = (field - field.mean()) / field.std()
        field = self.amplitude * field
        return field.T  # convert to [Ny, Nx] for image processing conventions
    
    def resample(
        self,
        key: jax.Array
    ) -> jnp.ndarray: # B, 1, Ny, Nx
        keys = jax.random.split(key, self.num_u_samples)
        samples = jax.vmap(self.sample_one_u)(keys)  # B, Ny, Nx
        samples = rearrange(samples, "B Ny Nx -> B 1 Ny Nx")
        return samples
    
    
class DataFactory:
    
    def __init__(
        self,
        func_sampler: FunctionSampler,
        coord_sampler: CoordSampler,
    ):
        self.func_sampler = func_sampler
        self.coord_sampler = coord_sampler
        
    def get_batch(
        self,
        key: jax.Array
    ):
        key_func, key_coord = jax.random.split(key)
        u_samples = self.func_sampler.resample(key_func)
        pde_samples = self.coord_sampler.resample(key_coord)
        
        return u_samples, pde_samples
    
    
if __name__ == "__main__":
    func_sampler = FunctionSampler(lx=1.0, ly=1.0)
    coord_sampler = CoordSampler()
    data_factory = DataFactory(func_sampler, coord_sampler)
    
    key = jax.random.PRNGKey(0)
    u_samples, coord_samples = data_factory.get_batch(key)
    pde_samples = coord_samples["pde"]
    
    print("u_samples shape:", u_samples.shape)
    print("pde_samples shape:", pde_samples.shape)
    
    import matplotlib.pyplot as plt
    x_coord = pde_samples[:, 0]
    y_coord = pde_samples[:, 1]
    t_coord = pde_samples[:, 2]
    u = u_samples[0, 0] # first sample, u component  Nx, Ny
    v = u_samples[0, 1] # first sample, v component  Nx, Ny

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.imshow(u, extent=(0, 1, 0, 1), origin='lower', cmap="RdBu_r")
    ax.set_title('Sampled u field')
    
    ax = axes[1]
    ax.imshow(v, extent=(0, 1, 0, 1), origin='lower', cmap="RdBu_r")
    ax.set_title('Sampled v field')
    
    fig.tight_layout()
    fig.savefig("tmp/sampled_fields.png")
        
        
        