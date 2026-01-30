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
        num_ic_samples: int = 1024,
    ):
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.num_pde_samples = num_pde_samples
        self.num_ic_samples = num_ic_samples
        
        
    def sample_pde(self, key: jax.Array) -> jnp.ndarray:
        """Sample points within the spatio-temporal domain using LHS."""
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, t_max = self.temporal_domain

        mins = jnp.array([x_min, y_min, t_min])
        maxs = jnp.array([x_max, y_max, t_max])
        pde_points = lhs_sampling(mins, maxs, self.num_pde_samples, key)

        return pde_points
    
    def sample_ic(self, key: jax.Array) -> jnp.ndarray:
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, _ = self.temporal_domain

        mins = jnp.array([x_min, y_min])
        maxs = jnp.array([x_max, y_max])
        ic_points_xy = lhs_sampling(
            mins, maxs, self.num_ic_samples, key
        )  # shape (num_ic_samples, 2)
        t_ic = jnp.full((self.num_ic_samples, 1), t_min)  # shape (num_ic_samples, 1)
        ic_points = jnp.hstack([ic_points_xy, t_ic])

        return ic_points
    
    def resample(
        self,
        key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        pde_samples = self.sample_pde(key)
        ic_samples = self.sample_ic(key)
        
        return {
            "pde": pde_samples,
            "ic_ut": ic_samples
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
        return field.T # transpose to Ny, Nx
    
    def resample(
        self,
        key: jax.Array
    ) -> jnp.ndarray: # B, 1, Ny, Nx
        keys = jax.random.split(key, self.num_u_samples)
        samples = jax.vmap(self.sample_one_u)(keys)  # B, Ny, Nx
        samples = rearrange(samples, "B Nx Ny -> B 1 Nx Ny")  # B, C=1, Ny, Nx
        return samples
    
    
class DataFactory:
    
    def __init__(
        self,
        func_sampler: FunctionSampler,
        coord_sampler: CoordSampler,
    ):
        self.func_sampler_u = func_sampler
        self.coord_sampler = coord_sampler
        
    def get_batch(
        self,
        key: jax.Array
    ):
        key_func, key_coord = jax.random.split(key, 2)
        u_samples = self.func_sampler_u.resample(key_func)
        coords_samples = self.coord_sampler.resample(key_coord)

        return u_samples, coords_samples
    
    
if __name__ == "__main__":
    func_sampler_u = FunctionSampler(lx=1.0, ly=1.0, length_scale=0.1, amplitude=0.2)
    # func_sampler_c = FunctionSampler(lx=1.0, ly=1.0, length_scale=0.5, amplitude=0.2)
    
    coord_sampler = CoordSampler()
    data_factory = DataFactory(func_sampler_u, coord_sampler)
    
    key = jax.random.PRNGKey(0)
    u_samples, coords_samples = data_factory.get_batch(key)
    pde_samples = coords_samples["pde"]
    ic_samples = coords_samples["ic_ut"]
    
    print("u_samples shape:", u_samples.shape)

    
    import matplotlib.pyplot as plt
    x_coord = pde_samples[:, 0]
    y_coord = pde_samples[:, 1]
    t_coord = pde_samples[:, 2]
    u = u_samples[0, 0] # first sample, u component  Nx, Ny

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(u, extent=(0, 1, 0, 1), origin='lower', cmap="RdBu_r")
    ax.set_title('Sampled u field')

    
    fig.tight_layout()
    fig.savefig("tmp/sampled_fields.png")
        
        
        