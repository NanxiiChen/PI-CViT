from typing import Tuple, Dict
import argparse
import os
import time
from pprint import pprint

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
from models.utils import apply_overrides

from .configs import load_configs
from .losses import Losses
from .sample import CoordSampler, FunctionSampler
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
    weight_coef = jnp.array([1.0]*5),
    active_losses: Tuple[str] = ("loss_momentum", "loss_continuity", 
                                 "loss_bc_walls", "loss_bc_lid", "loss_bc_pressure"),
    **kwargs,
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
    arg_parser.add_argument(
        "--set",
        action="append",
        help="Override configuration values",
        default=[],
    )
    args = arg_parser.parse_args()
    configs_raw = load_configs(args.configs)
    configs = apply_overrides(configs_raw, args.set, strict=False)
    pprint(vars(configs), sort_dicts=False)
    
    save_dir = configs.save_dir + time.strftime("/%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    
    if configs.use_multi_gpu:
        devices = jax.devices()
        num_devices = len(devices)
        print(f"Number of devices: {num_devices}, devices: {devices}")
        mesh = Mesh(devices, axis_names=("batch",))
        # coords and model are replicated across devices
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        # split Re (`u` variables) data across devices
        # u shape: (B, 1)
        data_sharding = data_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    
    func_sampler = FunctionSampler(
        num_u_samples=configs.num_u_samples,
    )
    coord_sampler = CoordSampler(
        spatial_domain=configs.spatial_domain,
        num_pde_samples=configs.num_pde_samples,
        num_bc_samples=configs.num_bc_samples,
    )

    subkey, key = jax.random.split(key)
    model_params = configs.model_params
    if configs.model_name == "cvit":
        from .simplified_cvit import CViT
        model = CViT(
            subkey,
            **model_params,
        )
    elif configs.model_name == "deeponet":
        from .simplified_deeponet import DeepONet
        model = DeepONet(
            subkey,
            **model_params,
        )
        
    ckpt_path = configs.ckpt
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)
    
    if configs.use_multi_gpu:
        model = eqx.filter_shard(model, replicated_sharding)
        
    losses = Losses()
    loss_fn = losses.loss_fn
    active_loss_names = configs.active_loss_names
    active_losses = tuple(f"loss_{name}" for name in active_loss_names)

    if configs.optimizer_name == "soap":
        base_tx = optax.chain(
            optax.clip_by_global_norm(configs.max_grad_norm),
            soap(
                learning_rate=configs.initial_lr,
                b1=0.0,
                b2=0.95,
                precondition_frequency=5,
            )
        )

        optimizer = optax.contrib.schedule_free(
            base_tx,
            learning_rate=configs.initial_lr,
            b1=0.95,
        )
    else:
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
    coords = {
        "pde": None,
        "bc_walls": None,
        "bc_lid": None
    }
    
    def compute_reynolds_range_linear(
        epoch,
        initial_range,
        max_reynolds,
        warm_up_epochs,
        reach_max_epoch,   # 关键参数
    ):
        min_re, init_max_re = initial_range

        if epoch < warm_up_epochs:
            return initial_range

        if epoch >= reach_max_epoch:
            return (min_re, max_reynolds)

        # 实际参与增长的 epoch 数
        effective_epoch = epoch - warm_up_epochs
        total_growth_epochs = reach_max_epoch - warm_up_epochs

        # 线性插值
        alpha = effective_epoch / total_growth_epochs
        new_max_re = init_max_re + alpha * (max_reynolds - init_max_re)

        return (min_re, int(new_max_re))
        
    for epoch in range(0, epochs):

        cur_reynolds_range = compute_reynolds_range_linear(
            epoch,
            configs.re_range_initial,  # (100, 200)
            configs.re_range[1],       # 2000
            configs.warmup_epochs,
            reach_max_epoch=configs.reach_max_re_epoch
        )
        
        cur_reynolds_range_normed = (
            configs.normalize_re(cur_reynolds_range[0]),
            configs.normalize_re(cur_reynolds_range[1])
        )

        if epoch % configs.resample_coord_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            coords = coord_sampler.resample(subkey)
            if configs.use_multi_gpu:
                coords = jax.tree.map(
                    lambda x: jax.device_put(x, replicated_sharding), coords
                ) # put coords to devices
            
        if epoch % configs.resample_u_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            batch_u = func_sampler.resample(
                subkey,
                u_range=cur_reynolds_range_normed,
            )
            if configs.use_multi_gpu:
                batch_u = jax.device_put(batch_u, data_sharding)
            # print("Current Reynolds number range:", cur_reynolds_range)
            # print("Resampled training data, min u: Epoch", epoch,   
            #       jnp.min(configs.denormalize_re(batch_u)), 
            #       "max u:", jnp.max(configs.denormalize_re(batch_u)))
            
        if epoch % configs.test_every == 0:
            fig, l2 = evaluate_model(
                model,
                configs.evaluate_on_re,
                configs.data_dir,
                configs
            )
            writer.add_figure("eval/u_pred_vs_ref", fig, epoch)
            writer.add_scalar("eval/l2_error", l2, epoch)
            plt.close(fig)
            
        weight_coef = jnp.array([1.0, 1.0, 3.0, 3.0, 1.0]) \
            if epoch < configs.warmup_epochs else jnp.array([1.0]*len(active_losses))
        model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
            model,
            loss_fn,
            opt_state,
            optimizer,
            batch_u,
            coords,
            configs,
            last_weights,
            alpha_w=configs.alpha_w,
            weight_coef=weight_coef,
            active_losses=active_losses,
        )
        last_weights = weights
        
        # if epoch % configs.test_every == 0:
        #     reynolds = aux_vars.get("re_vals", None)
        #     print(f"Reynolds numbers in the batch: min {jnp.min(reynolds):.4e}, max {jnp.max(reynolds):.4e}")
        
        if epoch % configs.log_every == 0:
            print(
                f"Epoch {epoch}/{epochs}, "
                f"Each loss: {', '.join([f'{lv:.4e}' for lv in loss_values])}, "
                f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}, "
                f"Current Re range: {cur_reynolds_range}, "
            )

            writer.add_scalar("loss/total", total_loss.item(), epoch)
            writer.add_scalar("info/current_reynold_max", cur_reynolds_range[1], epoch)
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