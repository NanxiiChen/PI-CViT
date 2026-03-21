
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
from einops import rearrange

from models import get_optimizer
from models.causal import CausalWeightor
from models.utils import apply_overrides

from .configs import load_configs
from .losses import Losses
from .sample import CoordSampler, DataFactory, FunctionSampler
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




def sample_points(key, t, y, x, num_samples,):
    T = t.shape[0]
    H = y.shape[0]
    W = x.shape[0]
    total = T * H * W
    
    # sample num_samples coordinates from the spatio-temporal grid of size T x H x W for each input function
    idx = jax.random.randint(key, (num_samples,), 0, total)
    return  jnp.unravel_index(idx, (T, H, W)) # (num_samples,)




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
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        # split `u` to different devices along the batch dimension
        # u shape: B, C, H, W
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None, None, None))
        data_sharding_ref = NamedSharding(mesh, PartitionSpec("batch", None, None))
    
    
    func_sampler = FunctionSampler(
        lx=configs.lx, ly=configs.ly, 
        length_scale=configs.length_scale,
        amplitude=configs.amplitude,
        grid_size=configs.model_params["grid_size"],
        num_u_samples=configs.batch_size
    )
    coord_sampler = CoordSampler(
        spatial_domain=configs.spatial_domain,
        temporal_domain=configs.temporal_domain,
        num_pde_samples=configs.num_samples,
        num_ic_samples=configs.num_ic_samples,
    )
    
    subkey, key = jax.random.split(key)
    model_params = configs.model_params
    if configs.model_name == "cvit":
        from .periodic_cvit import PeriodicCViT
        model = PeriodicCViT(
            subkey,
            lx=configs.lx,
            ly=configs.ly,
            **model_params,
        )
        
    elif configs.model_name == "deeponet":
        from .periodic_deeponet import PeriodicDeepONet
        model = PeriodicDeepONet(
            subkey,
            lx=configs.lx,
            ly=configs.ly,
            **model_params,
        )
    else:
        raise ValueError(f"Unsupported model name: {configs.model_name}")
        
    
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
    
    causal_eps = jnp.array(configs.causality_params["initial_eps"])
    loss_fn = losses.loss_fn
    active_loss_names = configs.active_loss_names
    active_losses = tuple(f"loss_{name}" for name in active_loss_names)
    
    optimizer = get_optimizer(
        optimizer_name=configs.optimizer_name,
        init_value=configs.initial_lr,
        transition_steps=configs.decay_every,
        decay_rate=configs.decay_rate,
        staircase=False,
        end_value=configs.min_lr,
        b1=0.90,
        b2=0.90,
        precondition_frequency=5,
        weight_decay=0.0,
        max_grad_norm=configs.max_grad_norm,
    )
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    last_weights = jnp.array([1.0] * len(active_losses)) / len(active_losses)
    if configs.use_multi_gpu:
        opt_state = eqx.filter_shard(opt_state, replicated_sharding)
        last_weights = jax.device_put(last_weights, replicated_sharding)
        
    
    dataset_size = configs.dataset_size
    print("Dataset_size:", dataset_size)
    batch_size = configs.batch_size
    batch_size = min(batch_size, dataset_size)
    num_batches = dataset_size // batch_size
    
    train_data = jnp.load(configs.train_data_path) 
    x = train_data["x"]
    y = train_data["y"]
    t = train_data["times"]
    solutions = train_data["solutions"] # (total_num, T, C, H, W)
    
    # we select a subset of trajectories from solutions
    total_num = solutions.shape[0] # each ablitation model is trained on a subset of the full dataset, e.g., 64 trajectories out of 256 trajectories
    subkey, key = jax.random.split(key)
    selected_idx = jax.random.choice(subkey, total_num, (dataset_size,), replace=False)
    solutions = solutions[selected_idx] # (dataset_size, T, C, H, W)
    
    
    num_train_step = 0
    total_train_step = configs.total_train_step
    coords = {
        "pde": None,
        "ic_uv": None,
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
                    weight_coef = jnp.array([1.0, 1.0, 1.0, 3.0, 3.0])
                else:
                    weight_coef = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
                    
            if num_train_step % configs.test_every == 0:
                eval_key, key = jax.random.split(key)
                fig, l2 = evaluate_model(
                    model,
                    configs.target_ts,
                    configs.data_dir,
                    configs.Lc,
                    configs.Tc,
                    eval_key
                )
                writer.add_figure("eval/u_pred_vs_ref", fig, num_train_step)
                writer.add_scalar("eval/l2_error", l2, num_train_step)
                plt.close(fig)
                print(f"Epoch {num_train_step}/{total_train_step}, L2 error: {l2:.4e}")
                
            start_batch = batch_idx * batch_size
            end_batch = start_batch + batch_size
            batch_solutions = solutions[start_batch:end_batch] # (batch_size, T, C, H, W)
            
            
            if (num_train_step % configs.resample_every == 0) or (num_train_step >= configs.warmup_steps):
                # sampling for data loss
                subkey, key = jax.random.split(key)
                t_idx, y_idx, x_idx = sample_points(
                    subkey, t, y, x,
                    num_samples=configs.num_samples
                ) # (num_samples,), (num_samples, 2)
                t_data = t[t_idx] # (num_samples,)
                x_data = jnp.stack([x[x_idx], y[y_idx]], axis=-1) # (num_samples, 2)
                t_data = t_data[:, None] # (num_samples, 1)
                data_coords = jnp.concatenate([x_data, t_data], axis=-1) # (num_samples, 3)
                coords["data"] = data_coords
                
                u_data = batch_solutions[:, 0, ...] # (batch_size, C, H, W)
                u_data = u_data[:, 0:1, ...] # we only use the first channel of u for data loss, (batch_size, 1, H, W)
                batch_solutions_trans = rearrange(batch_solutions, "b t c h w -> b t h w c") # (batch_size, T, H, W, C)
                ref_data = batch_solutions_trans[:, t_idx, y_idx, x_idx, :] # (batch_size, num_samples, C) the reference solution at the sampled coordinates
            
                if num_train_step % 5 == 0:
                    # resample u for the PDE loss every 5 steps
                    subkey, key = jax.random.split(key)
                    u_pde = func_sampler.resample(subkey)    
                
                subkey, key = jax.random.split(key)
                pde_and_ic_coords = coord_sampler.resample(subkey)
                if configs.physics_on_data:
                    # we evaluate pde loss on the sampled data points
                    pde_coords = data_coords
                    ic_uv_coords = pde_and_ic_coords["ic_uv"]
                else:
                    # we sample another set of coordinates for pde loss
                    pde_coords = pde_and_ic_coords["pde"]
                    ic_uv_coords = pde_and_ic_coords["ic_uv"]
                coords["pde"] = pde_coords
                coords["ic_uv"] = ic_uv_coords
                        
                            
                if configs.use_multi_gpu:
                    coords = jax.tree.map(
                        lambda x: jax.device_put(x, replicated_sharding),
                        coords
                    )
                    u_data = jax.device_put(u_data, data_sharding)
                    ref_data = jax.device_put(ref_data, data_sharding_ref)
                    u_pde = jax.device_put(u_pde, data_sharding)
                        
            model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
                model,
                loss_fn,
                opt_state,
                optimizer,
                u_pde,
                coords,
                configs,
                last_weights,
                configs.alpha_w,
                weight_coef=weight_coef,
                active_losses=active_losses,
                causal_eps=causal_eps,
                sol_ref=ref_data,
                u_data=u_data,
            )
            last_weights = weights
            
            if configs.use_causality and (
                "momentum" in active_loss_names or "continuity" in active_loss_names
            ):
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
                
                if num_train_step % configs.test_every == 0:
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
                    writer.add_figure("causality/causal_info_momentum", causal_fig_momentum, num_train_step)
                    writer.add_figure("causality/causal_info_continuity", causal_fig_continuity, num_train_step)
                    plt.close(causal_fig_momentum)
                    plt.close(causal_fig_continuity)


            if num_train_step % configs.log_every == 0:
                print(
                    f"Epoch {num_train_step}/{total_train_step}, "
                    f"Each loss: {', '.join([f'{lv:.4e}' for lv in loss_values])}, "
                    f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}, "
                )

                writer.add_scalar("loss/total", total_loss.item(), num_train_step)
                for i, lv in enumerate(loss_values):
                    writer.add_scalar(f"loss/loss_{active_loss_names[i]}", lv.item(), num_train_step)
                for i, w in enumerate(weights):
                    writer.add_scalar(f"weight/weight_{active_loss_names[i]}", w.item(), num_train_step)

                writer.flush()
                
            num_train_step += 1
            
            
if __name__ == "__main__":
    main()
