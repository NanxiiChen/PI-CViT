import argparse
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# from .configs import Configs
from models import get_model

from .configs import load_configs
from .losses import Losses
from .sample import CoordSampler, DataFactory, FunctionSampler


def ic_fn(a, b, theta, x, y, epsilon):
    # a, b, theta: scalars
    x_rot = x * jnp.cos(theta) + y * jnp.sin(theta)
    y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta)

    term = jnp.sqrt((x_rot / a)**2 + (y_rot / b)**2) 
    scale = 2 * a * b / (a + b)
    dist = scale * (1.0 - term)

    u = jnp.tanh(dist / (jnp.sqrt(2) * epsilon)) # shape (H, W) or scalar
    return u


@eqx.filter_jit
def train_step(
    model,
    loss_fn,
    state,
    optimizer,
    batch_u,
    batch_params,
    x_coords,
    t_coords,
    cfg,
    ic_fn,
    **kwargs
):
    # batch_u: (B, 1, H, W)
    # batch_params: (B, 3)
    # x_coords: (N_query, 2)
    # t_coords: (N_query, 1)
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model, batch_u, batch_params, x_coords, t_coords, cfg, ic_fn, **kwargs
    )
    updates, new_state = optimizer.update(total_grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, total_loss, losses, weights, aux_vars


def main():
    key = jax.random.PRNGKey(0)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--configs",
        type=str,
        default="train_debug",
        help="Configuration file for training",
    )
    args = arg_parser.parse_args()
    configs = load_configs(args.configs)

    # Data preparation
    func_sampler = FunctionSampler(
        a_range=configs.a_range,
        b_range=configs.b_range,
        theta_range=configs.theta_range,
        spatial_domain=configs.spatial_domain_phys,
        grid_size=configs.model_params["grid_size"],
        num_u_samples=configs.num_u_samples,
    )
    coord_sampler = CoordSampler(
        spatial_domain=configs.spatial_domain,
        temporal_domain=configs.temporal_domain,
        num_ic_samples=configs.num_ic_samples,
        num_pde_samples=configs.num_pde_samples,
    )
    
    data_factory = DataFactory(
        func_sampler=func_sampler,
        coord_sampler=coord_sampler,
    )
    
    
    subkey, key = jax.random.split(key)
    batch_u, batch_params, pde_coords, ic_coords = data_factory.get_batch(subkey, epsilon=configs.epsilon)
    
    subkey, key = jax.random.split(key)
    model_params = configs.model_params
    model = get_model(
        model_name=configs.model_name,
        key=subkey,
        **model_params,
    )
    print(model)
    losses = Losses()
    loss_fn = losses.loss_fn
    
    loss = loss_fn(
        model, batch_u, batch_params, pde_coords[:, :2], pde_coords[:, 2:], configs, ic_fn,
    )
    print("Initial loss:", loss[0])
    
    
    
    # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # axes = axes.flatten()
    # for i in range(16):
    #     ax = axes[i]
    #     ax.pcolormesh(
    #         func_sampler.coords[0],
    #         func_sampler.coords[1],
    #         batch_u[i, 0],
    #         shading="auto",
    #         rasterized=True,
    #     )
    #     ax.set_title(f'a={batch_params[i,0]:.1f}, b={batch_params[i,1]:.1f}, θ={batch_params[i,2]/jnp.pi*180:.2f}')
        
    # plt.tight_layout()
    # plt.savefig("tmp/sampled_initial_conditions.png", dpi=300)
    # plt.close()
    
    # fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": "3d"})
    # ax.scatter(
    #     pde_coords[:, 0],
    #     pde_coords[:, 1],
    #     pde_coords[:, 2],
    #     s=1,
    #     c="b",
    #     marker=".",
    #     label="PDE points",
    # )
    # ax.scatter(
    #     ic_coords[:, 0],
    #     ic_coords[:, 1],
    #     ic_coords[:, 2],
    #     s=1,
    #     c="r",
    #     marker=".",
    #     label="IC points",
    # )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("t")
    # ax.set_title("Sampled Coordinates")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig("tmp/sampled_coordinates.png", dpi=300)
    # plt.close()
    
    
    
if __name__ == "__main__":
    main()