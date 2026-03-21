"""
This script trains data-driven models as ablation studies for the purely physics-informed models. 
The training data is shaped as: (B, T, C, H, W), 
where B is the batch size (number of input functions),
T is the number of time steps,
C is the number of channels (1 for scalar functions),
H and W are the spatial dimensions.

Instead of sampling continuous coordinates in the spatio-temporal domain, 
we sample discrete coordinates corresponding to the grid points in the training data.
For each input function, we randomly sample a subset of the grid points at each time step to form the training coordinates:
For example, for T=100, H=64, W=64, we can sample 2048 coordinates from the spatial-temporal grid of size 100x64x64
"""

from typing import Tuple
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
    x_pde: jnp.ndarray,
    t_pde: jnp.ndarray,
    cfg: dict,
    last_weights: jnp.ndarray,
    alpha_w: float, 
    weight_coef: jnp.array = jnp.array([1.0, 1.0]),
    active_losses: Tuple[str] = ("loss_pde", "loss_ic", ),
    **kwargs
):
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model, batch_u, 
        x_pde, t_pde, cfg, 
        last_weights, alpha_w,
        weight_coef, active_losses, 
        **kwargs
    )
    total_grad = jax.tree.map(lambda x: jnp.nan_to_num(x), total_grad)
    updates, new_state = optimizer.update(total_grad, state, model)
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
        "--save_name",
        type=str,
        default=None,
        help="Name of the save directory for checkpoints and logs. Default to current timestamp.",
    )
    
    arg_parser.add_argument(
        "--dataset_size",
        type=int,
        default=None,
        help="Number of training samples (spatio-temporal coordinates) for each input function. Default to the value in configs.",
    )
    
    args = arg_parser.parse_args()
    configs = load_configs(args.configs)
    save_name = getattr(args, "save_name", time.strftime("%Y%m%d-%H%M%S"))
    if save_name is None:
        save_name = time.strftime("%Y%m%d-%H%M%S")
    print("save_name:", save_name)
    
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
    
    
    dataset_size = getattr(args, "dataset_size")
    if dataset_size is None:
        dataset_size = configs.dataset_size
    print("dataset_size:", dataset_size)
    # dataset_size = configs.dataset_size
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
    while num_train_step < total_train_step:
        for batch_idx in range(num_batches):
            
            # weight_coef = jnp.ones(len(active_losses))
            if "pde" not in active_loss_names:
                weight_coef = jnp.ones(len(active_losses))
            else:
                if num_train_step < configs.warmup_steps:
                    weight_coef = jnp.array([1.0, 0.2, 2.0]) # warm up with only data loss
                else:
                    weight_coef = jnp.array([1.0, 1.0, 1.0]) # then add the PDE loss
            
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
                writer.flush()
          
            start_batch = batch_idx * batch_size
            end_batch = start_batch + batch_size
            batch_solutions = solutions[start_batch:end_batch] # (batch_size, T, C, H, W)


            if num_train_step % configs.resample_every == 0 or num_train_step >= configs.warmup_steps:
                # sampling for data loss
                subkey, key = jax.random.split(key)
                t_idx, y_idx, x_idx = sample_points(
                    subkey, t, y, x,
                    num_samples=configs.num_samples
                ) # (num_samples,), (num_samples, 2)
                t_data = t[t_idx] # (num_samples,)
                x_data = jnp.stack([x[x_idx], y[y_idx]], axis=-1) # (num_samples, 2)
                t_data = t_data[:, None] # (num_samples, 1)

                u_data = batch_solutions[:, 0, ...] # (batch_size, C, H, W) the initial condition for each input function in the batch
                batch_solutions_trans = rearrange(batch_solutions, "b t c h w -> b t h w c") # (batch_size, T, H, W, C)
                ref_data = batch_solutions_trans[:, t_idx, y_idx, x_idx, :] # (batch_size, num_samples, C) the reference solution at the sampled coordinates
                if configs.use_multi_gpu:
                    t_data = jax.device_put(t_data, replicated_sharding)
                    x_data = jax.device_put(x_data, replicated_sharding)
                    u_data = jax.device_put(u_data, data_sharding)
                    ref_data = jax.device_put(ref_data, data_sharding_ref)                
            
                # sampling for pde loss
                subkey, key = jax.random.split(key)
                pde_coords = coord_sampler.resample(subkey)
                x_pde, t_pde = pde_coords[:, 0:2], pde_coords[:, 2:3]
                subkey, key = jax.random.split(key)
                u_pde = func_sampler.resample(subkey)
                if configs.use_multi_gpu:
                    t_pde = jax.device_put(t_pde, replicated_sharding)
                    x_pde = jax.device_put(x_pde, replicated_sharding)
                    u_pde = jax.device_put(u_pde, data_sharding)


            
            model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
                model, loss_fn, opt_state, optimizer, u_pde,
                x_pde, t_pde, configs, last_weights, configs.alpha_w,
                weight_coef=weight_coef,
                active_losses=active_losses,
                causal_eps=causal_eps,
                sol_ref=ref_data,
                x_data=x_data,
                t_data=t_data,
                u_data=u_data,
            )
            last_weights = weights
            
            if configs.use_causality and "pde" in active_loss_names:    
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
                
                if num_train_step % configs.test_every == 0:
                    fig = causal_weightor.plot_causal_info(
                        loss_chunks,
                        causal_weights,
                        causal_eps,
                    )
                    writer.add_figure("causal_weights", fig, num_train_step)
                    plt.close(fig)     
            
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

            
    
    