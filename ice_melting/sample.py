from typing import Tuple, List
import jax
import jax.numpy as jnp
from models.sample import lhs_sampling


class CoordSampler:
    """
    Sampler for generating coordinate points in spatial and temporal domains.
    Used for PDE residual points and Initial Condition (IC) points.
    """

    def __init__(
        self,
        spatial_domain: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-0.5, 0.5),
            (-0.5, 0.5),
        ),
        temporal_domain: Tuple[float, float] = (0.0, 3.0),
        num_pde_samples: int = 1024,
        num_ic_samples: int = 256,
    ):
        """
        Initialize the coordinate sampler.

        Args:
            spatial_domain: Tuple of (x_range, y_range) where each range is (min, max).
            temporal_domain: Tuple of (t_min, t_max).
            num_pde_samples: Number of points to sample for PDE residual.
            num_ic_samples: Number of points to sample for initial conditions.
        """
        # spatial is normalized to [-0.5,0.5]
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
        """Sample points at t = t_min for initial conditions."""
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

    def resample(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a new set of PDE and IC samples."""
        key_pde, key_ic = jax.random.split(key, 2)
        pde_samples = self.sample_pde(key_pde)
        ic_samples = self.sample_ic(key_ic)

        return pde_samples, ic_samples


class FunctionSampler:
    """
    Sampler for generating initial scalar fields based on random ellipse geometries.
    """
    def __init__(self,
                 a_range=(20,40), # actual lengths
                 b_range=(20,40), # actual lengths
                 theta_range=(0,jnp.pi),
                 spatial_domain=((-0.5, 0.5), (-0.5, 0.5)), # normalized
                 grid_size=(224,224),
                 Lc=100, # feature edge length
                 ):
        """
        Initialize the function sampler.

        Args:
            a_range: Range for the semi-major axis.
            b_range: Range for the semi-minor axis.
            theta_range: Range for the rotation angle.
            spatial_domain: Normalized spatial boundaries.
            grid_size: Resolution of the output grid (H, W).
            Lc: Characteristic length scale to map normalized coords to physical domain.
        """
        # Calculate epsilon based on FEM logic
        self.a_range = a_range
        self.b_range = b_range
        self.theta_range = theta_range
        coord_x = jnp.linspace(spatial_domain[0][0], spatial_domain[0][1], grid_size[1])
        coord_y = jnp.linspace(spatial_domain[1][0], spatial_domain[1][1], grid_size[0])
        xv, yv = jnp.meshgrid(coord_x, coord_y, indexing='xy')
        # formulate coords as (2, H, W)
        self.coords = jnp.stack([xv, yv], axis=0) * Lc  # scale to physical domain
        

    def sample_params(self, num_u_samples: int, key: jax.Array) -> jnp.ndarray:
        """Randomly sample ellipse parameters (a, b, theta) with a >= b."""
        keys = jax.random.split(key, 3)
        a_raw = jax.random.uniform(keys[0], (num_u_samples, 1), minval=self.a_range[0], maxval=self.a_range[1])
        b_raw = jax.random.uniform(keys[1], (num_u_samples, 1), minval=self.b_range[0], maxval=self.b_range[1])
        
        # Ensure a is the semi-major axis (a >= b) to remove geometric redundancy
        a = jnp.maximum(a_raw, b_raw)
        b = jnp.minimum(a_raw, b_raw)
        
        theta = jax.random.uniform(keys[2], (num_u_samples, 1), minval=self.theta_range[0], maxval=self.theta_range[1])
        return jnp.concatenate([a, b, theta], axis=-1) # shape (num_u_samples, 3)

    def evaluate(self, epsilon: float, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the scalar field u for given ellipse parameters on the fixed grid.
        
        Args:
            epsilon: Interface width parameter.
            params: Array of shape (B, 3) containing [a, b, theta] for each sample.
            
        Returns:
            u: Scalar field array of shape (B, 1, H, W).
        """
        
        def eval_one_sample(a, b, theta, x, y, epsilon):
            """Compute u for a single set of parameters."""
            # Coordinate rotation 
            x_rot = x * jnp.cos(theta) + y * jnp.sin(theta) # (H, W)
            y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta) # (H, W)

            # Ellipse distance calculation
            term = jnp.sqrt((x_rot / a)**2 + (y_rot / b)**2) 
            scale = 2 * a * b / (a + b)
            dist = scale * (1.0 - term)

            # Phase field representation using tanh
            u = jnp.tanh(dist / (jnp.sqrt(2) * epsilon)) # shape (H, W)
            return u[None, ...]  # shape (1, H, W)
        
        x, y = self.coords[0, :, :], self.coords[1, :, :]  # (H, W)
        a, b, theta = params[:, 0], params[:, 1], params[:, 2]  # each shape (B,)
        u = jax.vmap(eval_one_sample, in_axes=(0, 0, 0, None, None, None))(
            a, b, theta, x, y, epsilon) # shape (B, 1, H, W)
        return u


if __name__ == "__main__":
    # sample and plot
    import matplotlib.pyplot as plt
    fs = FunctionSampler()
    key = jax.random.PRNGKey(333)
    params = fs.sample_params(8, key)
    u = fs.evaluate(1.0, params)
    print(u.shape)
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    axes = axes.flatten()
    for i in range(8):
        ax = axes[i]
        # im = ax.contourf(fs.coords[0], fs.coords[1], u[i, 0], levels=50, cmap='RdBu')
        ax.pcolormesh(fs.coords[0], fs.coords[1], u[i, 0], shading='auto', cmap='RdBu', rasterized=True)
        ax.set_title(f'a={params[i,0]:.1f}, b={params[i,1]:.1f}, θ={params[i,2]/jnp.pi*180:.2f}')
    plt.savefig('sample_function_sampler.png')
    plt.close()
    
    cs = CoordSampler()
    key = jax.random.PRNGKey(1)
    pde_samples, ic_samples = cs.resample(key)
    print("PDE samples shape:", pde_samples.shape)
    print("IC samples shape:", ic_samples.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pde_samples[:,0], pde_samples[:,1], pde_samples[:,2], c='b', label='PDE Samples', s=1)
    ax.scatter(ic_samples[:,0], ic_samples[:,1], ic_samples[:,2], c='r', label='IC Samples', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    ax.legend()
    plt.savefig('sample_coord_sampler.png')