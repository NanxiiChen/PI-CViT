import argparse
import os
from functools import partial
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

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
    model: eqx.Module,
    loss_fn: Losses.loss_fn,
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_u: jnp.ndarray,
    batch_params: jnp.ndarray,
    pde_coords: jnp.ndarray,
    ic_coords: jnp.ndarray,
    cfg: dict,
    ic_fn: callable,
    **kwargs
):
    # batch_u: (B, 1, H, W)
    # batch_params: (B, 3)
    # pde_coords: (N_query=num_pde_samples, 3)
    # ic_coords: (N_query=num_ic_samples, 3)
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model, batch_u, batch_params, pde_coords, ic_coords, cfg, ic_fn, **kwargs
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

    save_dir = configs.save_dir + time.strftime("/%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

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
    model_params = configs.model_params
    model = get_model(
        model_name=configs.model_name,
        key=subkey,
        **model_params,
    )
    # print(model)
    losses = Losses()
    loss_fn = losses.loss_fn
    active_loss_names = ["pde", "ic"]

    scheduler = optax.exponential_decay(
        init_value=configs.initial_lr,
        transition_steps=configs.decay_every,
        decay_rate=configs.decay_rate,
        staircase=False,
        end_value=1e-5,
    )
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


    epochs = configs.num_epochs
    for epoch in range(epochs):
        subkey, key = jax.random.split(key)
        batch_u, batch_params, pde_coords, ic_coords =\
              data_factory.get_batch(subkey, epsilon=configs.epsilon)
        
        model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
            model,
            loss_fn,
            opt_state,
            optimizer,
            batch_u,
            batch_params,
            pde_coords,
            ic_coords,
            configs,
            ic_fn,
        )



        if epoch % configs.log_every == 0:
            print(
                f"Epoch {epoch}/{epochs}, "
                f"Each loss: {', '.join([f'{lv:.4e}' for lv in loss_values])}, "
                f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}, "
            )

            writer.add_scalar("loss/total", total_loss.item(), epoch)
            for i, lv in enumerate(loss_values):
                writer.add_scalar(f"loss/loss_{active_loss_names[i]}", lv.item(), epoch)
            for i, w in enumerate(weights):
                writer.add_scalar(f"weight/weight_{active_loss_names[i]}", w.item(), epoch)

            writer.flush()

        if epoch % configs.save_every == 0:
            eqx.tree_serialise_leaves(
                os.path.join(save_dir, f"model_epoch_{epoch}.eqx"),
                model,
            )
    
    writer.close()
    
if __name__ == "__main__":
    main()