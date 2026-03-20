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

from models import get_optimizer
from models.causal import CausalWeightor
from models.utils import apply_overrides

from .configs import load_configs
from .loss_spectral import Losses
from .sample import FunctionSampler
from .evaluator import evaluate_fno_model


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    loss_fn: Losses.loss_fn,
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_u: jnp.ndarray,
    cfg: dict,
    active_losses,
    last_weights: jnp.ndarray,
    alpha_w: float,
    weight_coef: jnp.ndarray,
    **kwargs
):
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model,
        batch_u,
        cfg,
        active_losses=active_losses,
        last_weights=last_weights,
        alpha_w=alpha_w,
        weight_coef=weight_coef,
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
        default="train_fno",
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

    save_dir = configs.save_dir + time.strftime("/%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    if configs.use_multi_gpu:
        devices = jax.devices()
        mesh = Mesh(devices, axis_names=("batch",))
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None, None, None))

    func_sampler = FunctionSampler(
        lx=configs.lx,
        ly=configs.ly,
        length_scale=configs.length_scale,
        amplitude=configs.amplitude,
        grid_size=(configs.Nx, configs.Ny),
        num_u_samples=configs.num_u_samples,
    )

    subkey, key = jax.random.split(key)
    from models.fno import FNO
    model = FNO(subkey, **configs.model_params)

    ckpt_path = configs.ckpt
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)

    causal_weightor = CausalWeightor(
        num_chunks=configs.causality_params["num_chunks"],
        t_range=configs.temporal_domain,
    )
    losses = Losses(causal_weightor=causal_weightor if configs.use_causality else None)
    loss_fn = losses.loss_fn

    causal_eps = jnp.array(configs.causality_params["initial_eps"])

    active_loss_names = getattr(configs, "active_loss_names", ["continuity", "momentum"])
    active_losses = tuple(
        n if n.startswith("loss_") else f"loss_{n}" for n in active_loss_names
    )
    alpha_w = float(getattr(configs, "alpha_w", 1.0))
    weight_coef = jnp.asarray(
        getattr(configs, "loss_weight_coef", [1.0] * len(active_losses)),
        dtype=jnp.float32
    )
    if weight_coef.shape[0] < len(active_losses):
        pad = jnp.ones((len(active_losses) - weight_coef.shape[0],), dtype=weight_coef.dtype)
        weight_coef = jnp.concatenate([weight_coef, pad], axis=0)
    else:
        weight_coef = weight_coef[:len(active_losses)]

    last_weights = jnp.ones((len(active_losses),), dtype=jnp.float32)

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
        weight_decay=0,
        max_grad_norm=configs.max_grad_norm,
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epochs = configs.num_epochs
    batch_u = None

    for epoch in range(epochs):
        if epoch % configs.resample_u_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            batch_u = func_sampler.resample(subkey)
            B, _, Nx, Ny = batch_u.shape
            uv0 = jnp.zeros((B, 2, Nx, Ny), dtype=batch_u.dtype)
            batch_u = jnp.concatenate([batch_u, uv0], axis=1)
            if configs.use_multi_gpu:
                batch_u = jax.device_put(batch_u, data_sharding)

        if epoch % configs.test_every == 0:
            eval_key, key = jax.random.split(key)
            fig, l2 = evaluate_fno_model(
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
            configs,
            active_losses,
            last_weights,
            alpha_w,
            weight_coef,
            causal_eps=causal_eps,
        )
        last_weights = jax.lax.stop_gradient(weights)

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

            new_eps = jnp.minimum(new_eps_momentum, new_eps_continuity)
            if abs(new_eps - causal_eps) > 1e-6:
                print(f"Update epsilon: {causal_eps:.4e} --> {new_eps:.4e}")
            causal_eps = new_eps

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
                f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}"
            )
            writer.add_scalar("loss/total", total_loss.item(), epoch)
            for i, lv in enumerate(loss_values):
                writer.add_scalar(f"loss/loss_{active_loss_names[i]}", lv.item(), epoch)
            for i, w in enumerate(weights):
                writer.add_scalar(f"weight/weight_{active_loss_names[i]}", w.item(), epoch)
            writer.flush()

        # if epoch % configs.save_every == 0:
        #     eqx.tree_serialise_leaves(
        #         os.path.join(save_dir, f"model_epoch_{epoch}.eqx"),
        #         model,
        #     )

    writer.close()


if __name__ == "__main__":
    main()
