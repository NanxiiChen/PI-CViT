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
from models.fno import FNO

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
    batch_u: jnp.ndarray,   # (B, C, Nx, Ny), C=1 for AC
    cfg,
    **kwargs,
):
    (total_loss, aux_vars), total_grad = loss_fn(model, batch_u, cfg, **kwargs)
    total_grad = jax.tree.map(lambda x: jnp.nan_to_num(x), total_grad)
    updates, new_state = optimizer.update(total_grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, total_loss, aux_vars


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
        "--optimizer_name",
        type=str,
        default=None,
        help="Optimizer name",
    )
    args = arg_parser.parse_args()
    configs = load_configs(args.configs)
    optimizer_name = args.optimizer_name if args.optimizer_name is not None else configs.optimizer_name

    save_dir = configs.save_dir + time.strftime("/%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    # optional multi-gpu sharding
    if getattr(configs, "use_multi_gpu", False):
        devices = jax.devices()
        mesh = Mesh(devices, axis_names=("batch",))
        # u shape: (B, C, Nx, Ny)
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None, None, None))
        print(f"Number of devices: {len(devices)}, devices: {devices}")

    # 仅采样函数场，训练时只用 batch_u
    # 若你的 FunctionSampler 参数名与 burgers 不同，请按你项目里的 sample.py 对齐
    func_sampler = FunctionSampler(
        a_range=configs.a_range,
        b_range=configs.b_range,
        theta_range=configs.theta_range,
        spatial_domain=getattr(configs, "spatial_domain_phys", configs.spatial_domain),
        grid_size=(configs.Nx, configs.Ny),
        num_u_samples=configs.num_u_samples,
    )

    # FNO
    subkey, key = jax.random.split(key)
    model = FNO(subkey, **configs.model_params)

    ckpt_path = getattr(configs, "ckpt", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)

    # losses
    causal_weightor = CausalWeightor(
        num_chunks=configs.causality_params["num_chunks"],
        t_range=configs.temporal_domain,
    )
    losses = Losses(
        causal_weightor=causal_weightor if configs.use_causality else None,
        time_scheme=getattr(configs, "time_scheme", "rk4"),
        spatial_scheme=getattr(configs, "spatial_scheme", "fd"),
    )
    loss_fn = losses.loss_fn
    causal_eps = jnp.array(configs.causality_params["initial_eps"])

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
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

    epochs = configs.num_epochs
    batch_u = None
    epsilon = getattr(configs, "epsilon", 1.0)

    for epoch in range(epochs):
        # 重采样 u0
        resample_every = getattr(configs, "resample_u_every", 1)
        warmup_epochs = getattr(configs, "warmup_epochs", 0)
        if (epoch % resample_every == 0) or (epoch >= warmup_epochs):
            key_params, key = jax.random.split(key)
            params = func_sampler.sample_params(key_params)         # (B, 3)
            batch_u = func_sampler.evaluate(epsilon, params)        # (B, 1, H, W)

            if getattr(configs, "use_multi_gpu", False):
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

        model, opt_state, total_loss, aux_vars = train_step(
            model,
            loss_fn,
            opt_state,
            optimizer,
            batch_u,
            configs,
            causal_eps=causal_eps,
        )

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
                f"Total Loss: {total_loss:.4e}, "
            )
            writer.add_scalar("loss/total", total_loss.item(), epoch)
            writer.flush()

        # if epoch % configs.save_every == 0:
        #     eqx.tree_serialise_leaves(
        #         os.path.join(save_dir, f"model_epoch_{epoch}.eqx"),
        #         model,
        #     )

    writer.close()


if __name__ == "__main__":
    main()