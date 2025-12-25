from typing import Tuple, List
import jax
import jax.numpy as jnp
from models.sample import lhs_sampling


class CoordSampler:

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
        # spatial is normalized to [-0.5,0.5]
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.num_pde_samples = num_pde_samples
        self.num_ic_samples = num_ic_samples

    def sample_pde(self, key) -> jnp.ndarray:
        x_min, x_max = self.spatial_domain[0]
        y_min, y_max = self.spatial_domain[1]
        t_min, t_max = self.temporal_domain

        mins = jnp.array([x_min, y_min, t_min])
        maxs = jnp.array([x_max, y_max, t_max])
        pde_points = lhs_sampling(mins, maxs, self.num_pde_samples, key)

        return pde_points

    def sample_ic(self, key):
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

    def resample(self, key) -> List[jnp.ndarray]:
        key_pde, key_ic = jax.random.split(key, 2)
        pde_samples = self.sample_pde(key_pde)
        ic_samples = self.sample_ic(key_ic)

        return pde_samples, ic_samples


class FunctionSampler:
    def __init__(self, epsilon: float, 
                 a_range=(20,40), # actual lengths
                 b_range=(20,40), # actual lengths
                 theta_range=(0,jnp.pi),
                 spatial_domain=((-0.5, 0.5), (-0.5, 0.5)), # normalized
                 grid_size=(224,224),
                 Lc=100, # feature edge length
                 ):
        # 依照 FEM 逻辑计算 epsilon
        self.epsilon = epsilon
        self.a_range = a_range
        self.b_range = b_range
        self.theta_range = theta_range
        coord_x = jnp.linspace(spatial_domain[0][0], spatial_domain[0][1], grid_size[0])
        coord_y = jnp.linspace(spatial_domain[1][0], spatial_domain[1][1], grid_size[1])
        xv, yv = jnp.meshgrid(coord_x, coord_y, indexing='ij')
        # formulate coords as (2, H, W)
        self.coords = jnp.stack([xv, yv], axis=0) * Lc  # scale to physical domain
        

    def sample_params(self, num_u_samples: int, key) -> jnp.ndarray:
        keys = jax.random.split(key, 3)
        a = jax.random.uniform(keys[0], (num_u_samples, 1), minval=self.a_range[0], maxval=self.a_range[1])
        b = jax.random.uniform(keys[1], (num_u_samples, 1), minval=self.b_range[0], maxval=self.b_range[1])
        theta = jax.random.uniform(keys[2], (num_u_samples, 1), minval=self.theta_range[0], maxval=self.theta_range[1])
        return jnp.concatenate([a, b, theta], axis=-1) # shape (num_u_samples, 3)

    def evaluate(self, params: jnp.ndarray) -> jnp.ndarray:
        # generage u on fixed grid coords with given ellipse params
        # coords will be fixed with shape (2, H, W)
        # params shape: (B, 3) where B is batch size
        
        def eval_one_sample(a, b, theta, x, y, epsilon):
        
            # 坐标旋转 
            x_rot = x * jnp.cos(theta) + y * jnp.sin(theta) # (H, W)
            y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta) # (H, W)

            # 椭圆项与距离计算
            term = jnp.sqrt((x_rot / a)**2 + (y_rot / b)**2) 
            scale = 2 * a * b / (a + b)
            dist = scale * (1.0 - term)

            return jnp.tanh(dist / (jnp.sqrt(2) * epsilon)) # shape (B, H, W)
        
        x, y = self.coords[0, :, :], self.coords[1, :, :]  # (H, W)
        a, b, theta = params[:, 0], params[:, 1], params[:, 2]  # each shape (B,)
        u = jax.vmap(eval_one_sample, in_axes=(0,0,0,None,None,None))(
            a, b, theta, x, y, self.epsilon) # shape (B, H, W)
        return u  # shape (B, H, W)



if __name__ == "__main__":
    # sample and plot
    import matplotlib.pyplot as plt
    fs = FunctionSampler(epsilon=1.0)
    key = jax.random.PRNGKey(111)
    params = fs.sample_params(8, key)
    u = fs.evaluate(params)
    print(u.shape)
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    axes = axes.flatten()
    for i in range(8):
        ax = axes[i]
        im = ax.contourf(fs.coords[0], fs.coords[1], u[i], levels=50, cmap='RdBu')
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