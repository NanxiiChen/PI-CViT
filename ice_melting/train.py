import argparse
import os
import time
from dataclasses import asdict

import equinox as eqx
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tensorboardX import SummaryWriter


# from .configs import Configs
from models import get_model, get_optimizer
from models.causal import CausalWeightor

from .configs import load_configs
from .losses import Losses
from .sample import CoordSampler, DataFactory, FunctionSampler
from .evaluator import evaluate_model

def ic_fn(a, b, theta, x, y, epsilon, Lc=None):
    # a, b, x, y, are expected to be the physical coordinates
    
    if Lc is not None:
        # indicate that x, y are normalized coordinates
        x = x * Lc
        y = y * Lc

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
    last_weights: jnp.ndarray,
    alpha_w: float,
    active_losses: tuple = ("loss_pde", "loss_ic", "loss_irr"),
    **kwargs 
    # kwargs includes causal_eps, 
    # it has beed converted to jax array outside to be traced in jit
):
    # batch_u: (B, 1, H, W)
    # batch_params: (B, 3)
    # pde_coords: (N_query=num_pde_samples, 3)
    # ic_coords: (N_query=num_ic_samples, 3)
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model, batch_u, batch_params, 
        pde_coords, ic_coords, cfg, 
        ic_fn, last_weights, alpha_w,
        active_losses, **kwargs
    )
    # nan to num
    total_grad = jax.tree.map(lambda x: jnp.nan_to_num(x), total_grad)
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
        num_rar_samples=configs.num_rar_samples,
        num_rar_pools=configs.num_rar_pools,
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

    ckpt_path = configs.ckpt
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)

    causal_weightor = CausalWeightor(
        num_chunks=configs.causality_params["num_chunks"],
        t_range=configs.temporal_domain,
    )

    losses = Losses(causal_weightor=causal_weightor)
    # !!! make `causal_eps` jax array, 
    # !!! so that it can be traced in jit
    # !!! otherwise, it will cause jit compilation every time when `causal_eps` is updated
    causal_eps = jnp.array(configs.causality_params["initial_eps"])
    loss_fn = losses.loss_fn
    active_loss_names = ("pde", "ic", "irr")
    active_losses = tuple(f"loss_{name}" for name in active_loss_names)

    # optimizer = optax.adam(scheduler)
    optimizer = get_optimizer(
        optimizer_name=configs.optimizer_name,
        init_value=configs.initial_lr,
        transition_steps=configs.decay_every,
        decay_rate=configs.decay_rate,
        staircase=False,
        end_value=configs.min_lr,
        b1=0.95,
        b2=0.95,
        precondition_frequency=5,
        weight_decay=1e-3,
        max_grad_norm=configs.max_grad_norm,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array)) 

    last_weights = jnp.array([1.0] * len(active_losses)) / len(active_losses)
    epochs = configs.num_epochs
    for epoch in range(epochs):
        subkey, key = jax.random.split(key)
        if epoch % configs.resample_every == 0:
            batch_u, batch_params, pde_coords, ic_coords =\
                data_factory.get_batch(subkey, model, losses.residual_pde, configs)
            

        if epoch % configs.test_every == 0:
            eval_key, key = jax.random.split(key)
            fig, l2 = evaluate_model(
                model,
                configs.target_ts,
                configs.data_dir,
                configs.Lc,
                configs.Tc,
                eval_key,
            )
            writer.add_figure("eval/u_pred_vs_ref", fig, epoch)
            writer.add_scalar("eval/l2_error", l2, epoch)
            plt.close(fig)
        
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
            last_weights,
            configs.alpha_w,
            active_losses=active_losses,
            causal_eps=causal_eps,
        )
        last_weights = weights

        if configs.use_causality:
            loss_chunks = aux_vars.get("loss_chunks", None)
            causal_weights = aux_vars.get("causal_weights", None)
            new_eps = causal_weightor.update_causal_eps(
                causal_weights,
                eps=causal_eps,
                max_eps=configs.causality_params["max_eps"],
                min_mean_weight=configs.causality_params["min_mean_weight"],
                max_min_weight=configs.causality_params["max_min_weight"],
                step_size=configs.causality_params["step_size"],
            )
            if abs(new_eps - causal_eps) > 1e-6:
                print(f"Update epsilon: {causal_eps:.4e} --> {new_eps:.4e}")
            # configs.causality_params.update(dict(eps=new_eps))
            causal_eps = new_eps
            

            if epoch % configs.test_every == 0:
                fig = causal_weightor.plot_causal_info(
                    loss_chunks,
                    causal_weights,
                    causal_eps,
                )
                writer.add_figure("causal_weights", fig, epoch)
                plt.close(fig)

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