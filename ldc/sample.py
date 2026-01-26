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
        num_pde_samples: int = 1024,
        num_bc_samples: int = 256,
        eps: float = 2e-2,
    ):
        self.spatial_domain = spatial_domain
        self.num_pde_samples = num_pde_samples
        self.num_bc_samples = num_bc_samples
        self.eps = eps
        
        
    def sample_pde(self, key: jax.Array) -> jnp.ndarray:
        """Sample points within the spatio-temporal domain using LHS."""
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]

        mins = jnp.array([x_min, y_min])
        maxs = jnp.array([x_max, y_max])
        pde_points = lhs_sampling(mins, maxs, 
                                  self.num_pde_samples, key)
        
        # replace the filtered points with safe points
        safe_point = jnp.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
                               dtype=pde_points.dtype)
        mask_invalid = ((pde_points[:, 0] > x_max - self.eps) 
                        & (pde_points[:, 1] > y_max - self.eps))
        mask_valid = ~mask_invalid
        pde_points = jnp.where(
            mask_valid[:, None], pde_points, safe_point[None, :]
        )
        
        return pde_points
    
    def sample_bc_walls(self, key: jax.Array) -> jnp.ndarray:
        """Sample points on the left, bottom, and right walls where `u=0, v=0`."""
        
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        k1, k2, k3 = jax.random.split(key, 3)
        
        # since the right-top and left-top corner is singular, we mask it out
        # y > y_max - self.eps
         
        # left wall (x = x_min)
        left_wall_y = lhs_sampling(
            jnp.array([y_min]), jnp.array([y_max - self.eps]), 
            self.num_bc_samples, k1
        )
        left_wall = jnp.concatenate(
            [jnp.full((self.num_bc_samples, 1), x_min), left_wall_y], axis=-1
        )
        
        # bottom wall (y = y_min)
        bottom_wall_x = lhs_sampling(
            jnp.array([x_min]), jnp.array([x_max]), 
            self.num_bc_samples, k2
        )
        bottom_wall = jnp.concatenate(
            [bottom_wall_x, jnp.full((self.num_bc_samples, 1), y_min)], axis=-1
        )
        
        # right wall (x = x_max)
        right_wall_y = lhs_sampling(
            jnp.array([y_min]), jnp.array([y_max - self.eps]), 
            self.num_bc_samples, k3
        )
        right_wall = jnp.concatenate(
            [jnp.full((self.num_bc_samples, 1), x_max), right_wall_y], axis=-1
        )
        bc_points = jnp.concatenate([left_wall, bottom_wall, right_wall], axis=0)
        return bc_points
    
    def sample_bc_lid(self, key: jax.Array) -> jnp.ndarray:
        """Sample points on the top wall with non-slip boundary condition."""
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        
        # top wall (y = y_max)
        top_wall_x = lhs_sampling(
            jnp.array([x_min + self.eps]), jnp.array([x_max - self.eps]), 
            self.num_bc_samples, key
        )
        top_wall = jnp.concatenate(
            [top_wall_x, jnp.full((self.num_bc_samples, 1), y_max)], axis=-1
        )
        return top_wall
        
    
    def resample(
        self,
        key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        pde_samples = self.sample_pde(key)
        bc_samples_walls = self.sample_bc_walls(key)
        bc_samples_lid = self.sample_bc_lid(key)
        
        return {
            "pde": pde_samples,
            "bc_walls": bc_samples_walls,
            "bc_lid": bc_samples_lid
        }
        
        
class FunctionSampler:
    
    def __init__(
        self,
        num_u_samples: int = 16,
    ):
        self.num_u_samples = num_u_samples
        
        
    def resample(
        self,
        key: jax.Array,
        u_range: Tuple[float, float] = (0.0, 1.0),
    ) -> jnp.ndarray:
        # We'll dynamically adjust the range of `u` during training
        # from small to large to stabilize training.
        # u represents the `nu` coefficient here, which is a constant scalar.
        u_min, u_max = u_range
        # u_samples = jax.random.uniform(
        #     key, (self.num_u_samples, 1), minval=u_min, maxval=u_max
        # )
        u_samples = lhs_sampling(
            jnp.array([u_min]), jnp.array([u_max]), 
            self.num_u_samples, key
        )
        return u_samples
    
    
    
class DataFactory:
    
    def __init__(
        self,
        coord_sampler: CoordSampler,
        function_sampler: FunctionSampler,
    ):
        self.coord_sampler = coord_sampler
        self.function_sampler = function_sampler
        
        
    def resample(
        self,
        key: jax.Array,
        u_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        k_coords, k_func = jax.random.split(key)
        coords = self.coord_sampler.resample(k_coords)
        u_samples = self.function_sampler.resample(k_func, u_range)
        return u_samples, coords
    
    
if __name__ == "__main__":
    func_sampler = FunctionSampler(num_u_samples=16)
    coord_sampler = CoordSampler()
    data_factory = DataFactory(coord_sampler, func_sampler)
    
    key = jax.random.PRNGKey(0)
    u_samples, coords_samples = data_factory.resample(key)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(
        coords_samples["pde"][:, 0], coords_samples["pde"][:, 1], 
        s=5, c='blue', label='PDE Points'
    )
    ax.scatter(
        coords_samples["bc_walls"][:, 0], coords_samples["bc_walls"][:, 1], 
        s=5, c='red', label='BC Walls'
    )
    ax.scatter(
        coords_samples["bc_lid"][:, 0], coords_samples["bc_lid"][:, 1], 
        s=5, c='green', label='BC Top'
    )
    ax.set_title('Sampled Coordinates')
    ax.legend()
    fig.savefig('tmp/sampled_coordinates.png', dpi=300)