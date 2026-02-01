from typing import Tuple, Dict
import argparse
import os
import time

import equinox as eqx
import jax
jax.config.update("jax_default_matmul_precision", "highest")
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tensorboardX import SummaryWriter

from models import get_optimizer
from models.soap import soap
from models.causal import CausalWeightor

from .configs import load_configs
from .losses import Losses
from .sample import CoordSampler, DataFactory, FunctionSampler
from .periodic_cvit import PeriodicCViT
from .evaluator import evaluate_model



@eqx.filter_jit
def train_step(
    model: eqx.Module,
    loss_fn: Losses.loss_fn,
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_u: jnp.ndarray,
    coords: Dict[str, jnp.ndarray],
    cfg: dict,
    last_weights: jnp.ndarray,
    alpha_w: float, 
    weight_coef: jnp.array = jnp.array([1.0, 1.0]),
    active_losses: Tuple[str] = ("loss_momentum", "loss_continuity",
                                 "loss_ic_h", "loss_ic_uv"),
    **kwargs
):
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model, batch_u, coords, cfg, 
        last_weights, alpha_w,
        weight_coef, active_losses, 
        **kwargs
    )
    total_grad = jax.tree.map(lambda x: jnp.nan_to_num(x), total_grad)
    params = eqx.filter(model, eqx.is_array)
    updates, new_state = optimizer.update(total_grad, state, params, is_training=True,)
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

    if configs.use_multi_gpu:
        devices = jax.devices()
        num_devices = len(devices)
        print(f"Number of devices: {num_devices}, devices: {devices}")
        mesh = Mesh(devices, axis_names=("batch",))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        # split `u` to different devices along the batch dimension
        # u shape: B, C, H, W
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None, None, None))
    
    func_sampler = FunctionSampler(
        lx=configs.lx, ly=configs.ly, 
        length_scale=configs.length_scale,
        amplitude=configs.amplitude,
        grid_size=configs.model_params["grid_size"],
        num_u_samples=configs.num_u_samples
    )
    coord_sampler = CoordSampler(
        spatial_domain=configs.spatial_domain,
        temporal_domain=configs.temporal_domain,
        num_pde_samples=configs.num_pde_samples,
    )
    
    subkey, key = jax.random.split(key)
    model_params = configs.model_params
    if configs.model_name == "cvit":
        
        model = PeriodicCViT(
            subkey,
            lx=configs.lx,
            ly=configs.ly,
            **model_params,
        )
        
    ckpt_path = configs.ckpt
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)
    
    if configs.use_multi_gpu:
        model = eqx.filter_shard(model, replicated_sharding)

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
    active_loss_names = configs.active_loss_names
    active_losses = tuple(f"loss_{name}" for name in active_loss_names)
    
    # base_tx = optax.chain(
    #     optax.clip_by_global_norm(configs.max_grad_norm),
    #     soap(
    #         learning_rate=configs.initial_lr,
    #         b1=0.95,
    #         b2=0.95,
    #         precondition_frequency=5,
    #     )
    # )

    # optimizer = optax.contrib.schedule_free(
    #     base_tx,
    #     learning_rate=configs.initial_lr,
    #     b1=0.95,
    # )
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
        weight_decay=1e-6,
        max_grad_norm=configs.max_grad_norm,
    )
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    last_weights = jnp.array([1.0] * len(active_losses)) / len(active_losses)
    if configs.use_multi_gpu:
        opt_state = eqx.filter_shard(opt_state, replicated_sharding)
        last_weights = jax.device_put(last_weights, replicated_sharding)
    
    epochs = configs.num_epochs
    batch_u = None
    coords = {"pde": None,}
    
    for epoch in range(0, epochs):
        if epoch % configs.resample_coord_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            coords = coord_sampler.resample(subkey)
            if configs.use_multi_gpu:
                coords = jax.tree.map(
                    lambda x: jax.device_put(x, replicated_sharding), coords
                ) # put coords to devices

            
        if epoch % configs.resample_u_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            batch_u = func_sampler.resample(subkey)
            
            if configs.use_multi_gpu:
                batch_u = jax.device_put(batch_u, data_sharding)
                
        if epoch % configs.test_every == 0:
            eval_key, key = jax.random.split(key)
            fig, l2 = evaluate_model(
                model,
                configs.target_ts,
                configs.data_dir,
                configs.Lc,
                configs.Tc,
                eval_key
            )
            writer.add_figure("eval/u_pred_vs_ref", fig, epoch)
            writer.add_scalar("eval/l2_error", l2, epoch)
            plt.close(fig)
            
            
        weight_coef = jnp.array([1.0, 1.0, 3.0, 3.0]) \
            if epoch < configs.warmup_epochs \
            else jnp.array([1.0] * len(active_losses))
        # weight_coef = jnp.array([1.0, 1.0, 5.0])
            
        model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
            model,
            loss_fn,
            opt_state,
            optimizer,
            batch_u,
            coords,
            configs,
            last_weights,
            configs.alpha_w,
            weight_coef=weight_coef,
            active_losses=active_losses,
            causal_eps=causal_eps,
        )
        last_weights = weights
        
        if configs.use_causality:
            loss_chunks_momentum = aux_vars.get("loss_chunks_momentum", None)
            causal_weights_momentum = aux_vars.get("causal_weights_momentum", None)
            loss_chunks_continuity = aux_vars.get("loss_chunks_continuity", None)
            causal_weights_continuity = aux_vars.get("causal_weights_continuity", None)
            new_eps_momentum = causal_weightor.update_causal_eps(
                causal_weights_momentum,
                eps=causal_eps,
                max_eps=configs.causality_params["max_eps"],
                min_mean_weight=configs.causality_params["min_mean_weight"],
                max_min_weight=configs.causality_params["max_min_weight"],
                step_size=configs.causality_params["step_size"],
            )
            new_eps_continuity = causal_weightor.update_causal_eps(
                causal_weights_continuity,
                eps=causal_eps,
                max_eps=configs.causality_params["max_eps"],
                min_mean_weight=configs.causality_params["min_mean_weight"],
                max_min_weight=configs.causality_params["max_min_weight"],
                step_size=configs.causality_params["step_size"],
            )
            if abs(jnp.minimum(new_eps_momentum, new_eps_continuity) - causal_eps) > 1e-6:
                print(f"Update epsilon: {causal_eps:.4e} --> {jnp.minimum(new_eps_momentum, new_eps_continuity):.4e}")
            # use the smaller one
            causal_eps = jnp.minimum(new_eps_momentum, new_eps_continuity)
            
            if epoch % configs.test_every == 0:
                causal_fig_momentum = causal_weightor.plot_causal_info(
                    loss_chunks=loss_chunks_momentum,
                    causal_weights=causal_weights_momentum,
                    eps=causal_eps,
                )
                causal_fig_continuity = causal_weightor.plot_causal_info(
                    loss_chunks=loss_chunks_continuity,
                    causal_weights=causal_weights_continuity,
                    eps=causal_eps,
                )
                writer.add_figure("causality/causal_info_momentum", causal_fig_momentum, epoch)
                writer.add_figure("causality/causal_info_continuity", causal_fig_continuity, epoch)
                plt.close(causal_fig_momentum)
                plt.close(causal_fig_continuity)
                
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