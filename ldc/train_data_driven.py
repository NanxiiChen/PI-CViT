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
from einops import rearrange

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



def sample_points(key, y, x, num_samples):
    """sample points from the given coordinates"""
    H = y.shape[0]
    W = x.shape[0]
    total = H * W
    
    idx = jax.random.randint(key, (num_samples,), 0, total)
    return jnp.unravel_index(idx, (H, W))


def main():
    key = jax.random.PRNGKey(0)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--configs",
        type=str,
        default="train_data_driven",
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
    save_name = getattr(configs, "save_name", None)
    if save_name is None:
        save_name = time.strftime("%Y%m%d-%H%M%S")
        
    save_dir = configs.save_dir + "/" + save_name
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
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
        # ref data: (B, num_samples, C=3)
        data_sharding_ref = NamedSharding(mesh, PartitionSpec("batch", None, None))
        
    func_sampler = FunctionSampler(
        num_u_samples=configs.batch_size,
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
        
    dataset_size = configs.dataset_size
    print(f"Dataset size: {dataset_size}")
    batch_size = configs.batch_size
    batch_size = min(batch_size, dataset_size)
    num_batches = dataset_size // batch_size

    train_data = jnp.load(configs.train_data_path) 
    x = train_data["x"]  # (H,)
    y = train_data["y"]  # (W,)
    solutions = train_data["solutions"] # (total_num, C, H, W) C=3 for u, v, p
    re_values_train = train_data["Re"] # (total_num,)
    
    total_num = solutions.shape[0]
    subkey, key = jax.random.split(key)
    selected_idx = jax.random.choice(subkey, total_num, (dataset_size,), replace=False)
    solutions = solutions[selected_idx] # (dataset_size, T, C, H, W)
    re_values_train = re_values_train[selected_idx] # (dataset_size,)
    
    
    num_train_step = 0
    total_train_step = configs.total_train_step
    coords = {
        "pde": None,
        "bc_walls": None,
        "bc_lid": None,
        "data": None,
    }
    
    while num_train_step < total_train_step:
        for batch_idx in range(num_batches):
            if ("momentum" not in active_loss_names) and ("continuity" not in active_loss_names):
                # only train with data
                weight_coef = jnp.ones(len(active_losses))
            else:
                # train with PDE losses
                if num_train_step < configs.warmup_steps:
                    weight_coef = jnp.array([1.0, 1.0, 1.0, 3.0, 3.0, 1.0])
                else:
                    weight_coef = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                    
            if num_train_step % configs.test_every == 0:
                fig, l2 = evaluate_model(
                    model,
                    configs.evaluate_on_re,
                    configs.data_dir,
                    configs
                )
                writer.add_figure("eval/u_pred_vs_ref", fig, num_train_step)
                writer.add_scalar("eval/l2_error", l2, num_train_step)
                plt.close(fig)
                print(f"Epoch {num_train_step}/{total_train_step}, L2 error: {l2:.4e}")
                
            start_batch = batch_idx * batch_size
            end_batch = start_batch + batch_size
            batch_solutions = solutions[start_batch:end_batch] # (batch_size, C, H, W)
            batch_re = re_values_train[start_batch:end_batch] # (batch_size,)
            
            if (num_train_step % configs.resample_every == 0) or (num_train_step >= configs.warmup_steps):
                # sampling for data loss
                subkey, key = jax.random.split(key)
                y_idx, x_idx = sample_points(
                    subkey, y, x,
                    num_samples=configs.num_pde_samples
                )
                x_data = jnp.stack([x[x_idx], y[y_idx]], axis=-1) # (num_samples, 2)
                coords["data"] = x_data
                
                u_data = configs.normalize_re(batch_re).reshape(-1,1) # (batch_size, 1)
                ref_data = rearrange(batch_solutions, "B C H W -> B H W C") # (B, H, W, C)
                ref_data = ref_data[:, y_idx, x_idx, :] # (batch_size, num_samples, C)
                
                # sample for PDE loss
                subkey, key = jax.random.split(key)
                cur_reynolds_range = configs.re_range
                cur_reynolds_range_normed = (
                    configs.normalize_re(cur_reynolds_range[0]),
                    configs.normalize_re(cur_reynolds_range[1]),
                )
                u_pde = func_sampler.resample(
                    subkey,
                    u_range=cur_reynolds_range_normed,
                )
                
                subkey, key = jax.random.split(key)
                pde_and_bc_coords = coord_sampler.resample(subkey)
                if configs.physics_on_data:
                    # Evaluate PDE loss on the labeled data points
                    pde_and_bc_coords["pde"] = x_data
                else:
                    # Sample PDE points separately from data points
                    # that is, keep using `pde_and_bc_coords` for PDE loss
                    # do nothing here
                    pass

                coords.update(pde_and_bc_coords)
                
                if configs.use_multi_gpu:
                    coords = jax.tree.map(
                        lambda x: jax.device_put(x, replicated_sharding),
                        coords
                    )
                    u_data = jax.device_put(u_data, data_sharding)
                    u_pde = jax.device_put(u_pde, data_sharding)
                    ref_data = jax.device_put(ref_data, data_sharding_ref)
                
            model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
                model,
                loss_fn,
                opt_state,
                optimizer,
                u_pde,
                coords,
                configs,
                last_weights,
                alpha_w=configs.alpha_w,
                weight_coef=weight_coef,
                active_losses=active_losses,
                sol_ref=ref_data,
                u_data=u_data,
            )
            last_weights = weights
            
            if num_train_step % configs.log_every == 0:
                print(
                    f"Epoch {num_train_step}/{total_train_step}, "
                    f"Each loss: {', '.join([f'{lv:.4e}' for lv in loss_values])}, "
                    f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}, "
                    f"Current Re range: {cur_reynolds_range}, "
                )

                writer.add_scalar("loss/total", total_loss.item(), num_train_step)
                writer.add_scalar("info/current_reynold_max", cur_reynolds_range[1], num_train_step)
                for i, lv in enumerate(loss_values):
                    writer.add_scalar(f"loss/loss_{active_loss_names[i]}", lv.item(), num_train_step)
                for i, w in enumerate(weights):
                    writer.add_scalar(f"weight/weight_{active_loss_names[i]}", w.item(), num_train_step)

                writer.flush()
                
            num_train_step += 1
                
    
if __name__ == "__main__":
    main()