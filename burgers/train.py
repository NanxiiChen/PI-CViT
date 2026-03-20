from typing import Tuple
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
    
    data_factory = DataFactory(
        func_sampler=func_sampler,
        coord_sampler=coord_sampler,
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
    # !!! make `causal_eps` jax array, 
    # !!! so that it can be traced in jit
    # !!! otherwise, it will cause jit compilation every time when `causal_eps` is updated
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
    
    epochs = configs.num_epochs
    
    batch_u = None
    x_pde = None
    t_pde = None
    for epoch in range(epochs):
        
        # at the beginning, we let the model train on a fixed set of collocation points for a few epochs
        # After that, we resample collocation points at every epoch
        if epoch % configs.resample_coord_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            pde_coords = coord_sampler.resample(subkey)
            # pde_coords are shared for each `u`
            pde_coords = jax.device_put(pde_coords, replicated_sharding) if configs.use_multi_gpu else pde_coords
            x_pde, t_pde = pde_coords[:, 0:2], pde_coords[:, 2:3]
            
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
            print(f"Epoch {epoch}/{epochs}, L2 error: {l2:.4e}")
            
        
        # at the warmup stage, we apply extra weighting to the IC loss
        # after warmup stage, all losses will be tuned by the GradNorm weights only
        weight_coef = jnp.array([0.5, 2.0]) \
            if epoch < configs.warmup_epochs \
            else jnp.array([1.0] * len(active_losses))
        model, opt_state, total_loss, loss_values, weights, aux_vars = train_step(
            model,
            loss_fn,
            opt_state,
            optimizer,
            batch_u,
            x_pde,
            t_pde,
            configs,
            last_weights,
            configs.alpha_w,
            weight_coef=weight_coef,
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

        if epoch % configs.save_every == 0 and configs.save_every > 0:
            eqx.tree_serialise_leaves(
                os.path.join(save_dir, f"model_epoch_{epoch}.eqx"),
                model,
            )
    
    writer.close()
                
    
if __name__ == "__main__":
    main()