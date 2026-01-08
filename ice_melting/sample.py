from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from models.sample import lhs_sampling, shifted_grid


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
        num_rar_samples: int = 0,
        num_rar_pools: int = 0,
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
        self.num_rar_samples = num_rar_samples
        self.num_rar_pools = num_rar_pools

    def sample_pde(self, key: jax.Array) -> jnp.ndarray:
        """Sample points within the spatio-temporal domain using LHS."""
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, t_max = self.temporal_domain

        mins = jnp.array([x_min, y_min, t_min])
        maxs = jnp.array([x_max, y_max, t_max])
        pde_points = lhs_sampling(mins, maxs, self.num_pde_samples, key)
        # pde_points = shifted_grid(
        #     mins, maxs, self.num_pde_samples, key
        # )  # shape (num_pde_samples, 3)

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

    def sample_rar(
        self,
        key: jax.Array,
        model: eqx.Module,
        residual_fn: callable,
        u: jnp.ndarray,
        params: jnp.ndarray,
        cfg: dict,
    ) -> jnp.ndarray:
        """Sample RAR points based on pde residuals."""
        # fns: pde residual function or other criteria functions
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, t_max = self.temporal_domain

        mins = jnp.array([x_min, y_min, t_min])
        maxs = jnp.array([x_max, y_max, t_max])
        pool_key, rar_key = jax.random.split(key)
        rar_pools = lhs_sampling(mins, maxs, self.num_rar_pools, pool_key)
        rar_pools_x = rar_pools[:, 0:2]
        rar_pools_t = rar_pools[:, 2:3]
        # Evaluate residuals at rar_pools
        residuals = jax.vmap(residual_fn, in_axes=(None, 0, 0, None, None, None))(
            model, u, params, rar_pools_x, rar_pools_t, cfg
        )  # B, N_query
        assert residuals.shape == (u.shape[0], self.num_rar_pools)

        residuals = jnp.mean(jnp.abs(residuals), axis=0)  # (N_rar_pools,)
        prob_dist = residuals / jnp.sum(residuals)
        rar_indices = jax.random.choice(
            rar_key,
            self.num_rar_pools,
            shape=(self.num_rar_samples,),
            p=prob_dist,
            replace=False,
        )
        rar_points = rar_pools[rar_indices, :]  # shape (num_rar_samples, 3)
        return rar_points

    def resample(
        self,
        key: jax.Array,
        model: eqx.Module,
        residual_fn: callable,
        u: jnp.ndarray,
        params: jnp.ndarray,
        cfg: dict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a new set of PDE and IC samples."""
        key_pde, key_rar, key_ic = jax.random.split(key, 3)
        pde_samples = self.sample_pde(key_pde)
        if self.num_rar_samples > 0 and self.num_rar_pools > 0:
            rar_samples = self.sample_rar(key_rar, model, residual_fn, u, params, cfg)
            pde_samples = jnp.concatenate([pde_samples, rar_samples], axis=0)
        ic_samples = self.sample_ic(key_ic)

        return pde_samples, ic_samples


class FunctionSampler:
    """
    Sampler for generating initial scalar fields based on random ellipse geometries.
    """

    def __init__(
        self,
        a_range=(20, 40),  # actual lengths
        b_range=(20, 40),  # actual lengths
        theta_range=(0, jnp.pi),
        spatial_domain=((-50, 50), (-50, 50)),  # actual lengths
        grid_size=(224, 224),
        num_u_samples=16,
    ):
        """
        Initialize the function sampler.

        Args:
            a_range: Range for the semi-major axis.
            b_range: Range for the semi-minor axis.
            theta_range: Range for the rotation angle.
            spatial_domain: Spatial boundaries of actual lengths. Note this is different with the normalized domain used in CoordSampler and the model.
            grid_size: Resolution of the output grid (H, W).

        """
        # Calculate epsilon based on FEM logic
        self.a_range = a_range
        self.b_range = b_range
        self.theta_range = theta_range
        coord_x = jnp.linspace(spatial_domain[0][0], spatial_domain[0][1], grid_size[1])
        coord_y = jnp.linspace(spatial_domain[1][0], spatial_domain[1][1], grid_size[0])
        xv, yv = jnp.meshgrid(coord_x, coord_y, indexing="xy")
        # formulate coords as (2, H, W)
        self.coords = jnp.stack([xv, yv], axis=0)
        self.num_u_samples = num_u_samples

    def sample_params(self, key: jax.Array) -> jnp.ndarray:
        """Use LHS to sample ellipse parameters (a, b, theta) with a >= b."""
        mins = jnp.array([self.a_range[0], self.b_range[0], self.theta_range[0]])
        maxs = jnp.array([self.a_range[1], self.b_range[1], self.theta_range[1]])

        samples = lhs_sampling(mins, maxs, self.num_u_samples, key)
        a_raw = samples[:, 0:1]
        b_raw = samples[:, 1:2]
        theta = samples[:, 2:3]

        # enforce a >= b
        a = jnp.maximum(a_raw, b_raw)
        b = jnp.minimum(a_raw, b_raw)

        return jnp.concatenate([a, b, theta], axis=-1)  # shape (num_u_samples, 3)

    def eval_one_sample(self, a, b, theta, x, y, epsilon):
        """Compute single `u` sample for given ellipse parameters on the grid or one coordinate."""
        # a, b, theta: scalars
        # Coordinate rotation
        x_rot = x * jnp.cos(theta) + y * jnp.sin(theta)  # (H, W) or scalar
        y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta)  # (H, W) or scaler

        # Ellipse distance calculation
        term = jnp.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
        scale = 2 * a * b / (a + b)
        dist = scale * (1.0 - term)  # shape (H, W) or scalar

        # Phase field representation using tanh
        u = jnp.tanh(dist / (jnp.sqrt(2) * epsilon))  # shape (H, W) or scalar
        return u[None, ...]  # shape (1, H, W) or (1,)

    def evaluate(self, epsilon: float, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the scalar field u for given ellipse parameters on the fixed grid.

        Args:
            epsilon: Interface width parameter.
            params: Array of shape (B, 3) containing [a, b, theta] for each sample.

        Returns:
            u: Scalar field array of shape (B, 1, H, W).
        """

        x, y = self.coords[0, :, :], self.coords[1, :, :]  # (H, W)
        a, b, theta = params[:, 0], params[:, 1], params[:, 2]  # each shape (B,)
        u = jax.vmap(self.eval_one_sample, in_axes=(0, 0, 0, None, None, None))(
            a, b, theta, x, y, epsilon
        )  # shape (B, 1, H, W)
        return u


class DataFactory:
    """
    Coordinates FunctionSampler and CoordSampler to produce training batches.
    """

    def __init__(self, func_sampler: FunctionSampler, coord_sampler: CoordSampler):
        self.func_sampler = func_sampler
        self.coord_sampler = coord_sampler

    def get_batch(self, key: jax.Array, 
                  model: eqx.Module,
                  residual_fn: callable,
                  cfg: dict,):
        key_u, key_coords = jax.random.split(key)
        epsilon = cfg.epsilon

        params = self.func_sampler.sample_params(key_u)  # shape (num_u, 3)
        u0_grid = self.func_sampler.evaluate(epsilon, params)  # shape (num_u, 1, H, W)

        pde_points, ic_points = self.coord_sampler.resample(
            key_coords, model, residual_fn, u0_grid, params, cfg
        )

        # return {
        #     "u0": u0_grid,           # input function field (num_u, 1, H, W)
        #     "params": params,       # ellipse parameters (num_u, 3)
        #     "pde_points": pde_points, # (N_pde, 3) -> [x, y, t]
        #     "ic_points": ic_points    # (N_ic, 3)  -> [x, y, 0]
        # }
        return u0_grid, params, pde_points, ic_points


if __name__ == "__main__":
    # sample and plot
    import matplotlib.pyplot as plt

    fs = FunctionSampler(grid_size=(64, 64), num_u_samples=9)
    key = jax.random.PRNGKey(0)
    params = fs.sample_params(key)
    epsilon = 6 * 100 / 63 / (2 * jnp.sqrt(2) * jnp.arctanh(0.9))
    u = fs.evaluate(epsilon, 
                    params)
    print(u.shape)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(9):
        ax = axes[i]
        # im = ax.contourf(fs.coords[0], fs.coords[1], u[i, 0], levels=50, cmap='RdBu')
        ax.pcolormesh(
            fs.coords[0],
            fs.coords[1],
            u[i, 0],
            shading="auto",
            cmap="coolwarm",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(0.5, 1.01, f"a={params[i,0]:.1f}, b={params[i,1]:.1f}, θ={params[i,2]/jnp.pi*180:.2f}°",
                ha="center", va="bottom", transform=ax.transAxes)
        # ax.set_title(
        #     f"a={params[i,0]:.1f}, b={params[i,1]:.1f}, θ={params[i,2]/jnp.pi*180:.2f}"
        # )
    plt.savefig("tmp/sample_function_sampler.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # cs = CoordSampler()
    # key = jax.random.PRNGKey(1)
    # pde_samples, ic_samples = cs.resample(key)
    # print("PDE samples shape:", pde_samples.shape)
    # print("IC samples shape:", ic_samples.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     pde_samples[:, 0],
    #     pde_samples[:, 1],
    #     pde_samples[:, 2],
    #     c="b",
    #     label="PDE Samples",
    #     s=1,
    # )
    # ax.scatter(
    #     ic_samples[:, 0],
    #     ic_samples[:, 1],
    #     ic_samples[:, 2],
    #     c="r",
    #     label="IC Samples",
    #     s=10,
    # )
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("T")
    # ax.legend()
    # plt.savefig("tmp/sample_coord_sampler.png")

    # factory = DataFactory(fs, cs)
    # batch = factory.get_batch(key, epsilon=1.0)

    # print(f"Input u0 shape: {batch['u0'].shape}")           # (16, 1, 224, 224)
    # print(f"PDE points shape: {batch['pde_points'].shape}") # (1024, 3)
