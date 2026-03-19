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
from models.fno import FNO2d

from .configs import load_configs
from .losses_spectral import Losses
from .sample import FunctionSampler
from .evaluator import evaluate_fno_model


def _expand_re_to_field(batch_re: jnp.ndarray, nx: int, ny: int) -> jnp.ndarray:
    """
    batch_re: (B,) or (B,1)
    return:   (B,1,Nx,Ny)
    """
    if batch_re.ndim == 2 and batch_re.shape[-1] == 1:
        batch_re = batch_re[:, 0]
    elif batch_re.ndim != 1:
        raise ValueError(f"batch_re must be (B,) or (B,1), got {batch_re.shape}")

    bsz = batch_re.shape[0]
    re_field = batch_re.reshape(bsz, 1, 1, 1)
    re_field = jnp.broadcast_to(re_field, (bsz, 1, nx, ny))
    return re_field


def compute_reynolds_range_linear(
    epoch: int,
    initial_range: Tuple[float, float],
    max_reynolds: float,
    warm_up_epochs: int,
    reach_max_epoch: int,
):
    min_re, init_max_re = initial_range

    if epoch < warm_up_epochs:
        return initial_range
    if epoch >= reach_max_epoch:
        return (min_re, max_reynolds)

    effective_epoch = epoch - warm_up_epochs
    total_growth_epochs = max(1, reach_max_epoch - warm_up_epochs)
    alpha = effective_epoch / total_growth_epochs
    new_max_re = init_max_re + alpha * (max_reynolds - init_max_re)
    return (min_re, float(new_max_re))


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    loss_fn: Losses.loss_fn,
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_u: jnp.ndarray,  # (B,1,Nx,Ny), normalized Re field
    cfg,
    active_losses,
    last_weights: jnp.ndarray,
    alpha_w: float,
    weight_coef: jnp.ndarray,
    **kwargs,
):
    (total_loss, (losses, weights, aux_vars)), total_grad = loss_fn(
        model,
        batch_u,
        cfg,
        active_losses=active_losses,
        last_weights=last_weights,
        alpha_w=alpha_w,
        weight_coef=weight_coef,
        **kwargs,
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
        "--optimizer_name",
        type=str,
        default=None,
        help="Optimizer name",
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save logs and checkpoints",
    )
    args = arg_parser.parse_args()
    configs = load_configs(args.configs)
    optimizer_name = args.optimizer_name if args.optimizer_name is not None else configs.optimizer_name

    save_dir = args.save_dir if args.save_dir is not None else configs.save_dir
    save_dir = save_dir + time.strftime("/%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    use_multi_gpu = bool(getattr(configs, "use_multi_gpu", False))
    if use_multi_gpu:
        devices = jax.devices()
        mesh = Mesh(devices, axis_names=("batch",))
        data_sharding = NamedSharding(mesh, PartitionSpec("batch", None, None, None))
        print(f"Number of devices: {len(devices)}, devices: {devices}")

    nx, ny = configs.Nx, configs.Ny

    # 采样 Re（标量），再扩展成 (B,1,Nx,Ny)
    func_sampler = FunctionSampler(
        num_u_samples=configs.num_u_samples,
    )

    subkey, key = jax.random.split(key)
    model = FNO2d(subkey, **configs.model_params)

    ckpt_path = getattr(configs, "ckpt", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Load model from checkpoint: {ckpt_path}")
        model = eqx.tree_deserialise_leaves(ckpt_path, model)

    losses = Losses()
    loss_fn = losses.loss_fn

    active_loss_names = getattr(
        configs,
        "active_loss_names",
        ["momentum", "continuity", "bc_walls", "bc_lid", "bc_pressure"],
    )
    active_losses = tuple(n if n.startswith("loss_") else f"loss_{n}" for n in active_loss_names)
    alpha_w = float(getattr(configs, "alpha_w", 1.0))

    # weight_coef = jnp.asarray(
    #     getattr(configs, "loss_weight_coef", [1.0] * len(active_losses)),
    #     dtype=jnp.float32,
    # )
    weight_coef = jnp.array([1.0, 1.0, 3.0, 5.0, 5.0], dtype=jnp.float32)

    last_weights = jnp.ones((len(active_losses),), dtype=jnp.float32)

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
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

    epochs = configs.num_epochs
    batch_u = None

    for epoch in range(epochs):
        # Re curriculum (physical range)
        cur_re_range = compute_reynolds_range_linear(
            epoch=epoch,
            initial_range=configs.re_range_initial,
            max_reynolds=configs.re_range[1],
            warm_up_epochs=configs.warmup_epochs,
            reach_max_epoch=configs.reach_max_re_epoch,
        )
        # normalize Re range
        cur_re_range_normed = (
            configs.normalize_re(cur_re_range[0]),
            configs.normalize_re(cur_re_range[1]),
        )

        if epoch % configs.resample_u_every == 0 or epoch >= configs.warmup_epochs:
            subkey, key = jax.random.split(key)
            batch_re = func_sampler.resample(
                subkey,
                u_range=cur_re_range_normed,
            )  # (B,) or (B,1), normalized
            batch_u = _expand_re_to_field(batch_re, nx=nx, ny=ny)  # (B,1,Nx,Ny)

            if use_multi_gpu:
                batch_u = jax.device_put(batch_u, data_sharding)

        if epoch % configs.test_every == 0:
            fig, l2 = evaluate_fno_model(
                model,
                configs.evaluate_on_re,
                configs.data_dir,
                configs,
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
            active_losses=active_losses,
            last_weights=last_weights,
            alpha_w=alpha_w,
            weight_coef=weight_coef,
        )
        last_weights = jax.lax.stop_gradient(weights)

        if epoch % configs.log_every == 0:
            print(
                f"Epoch {epoch}/{epochs}, "
                f"Each loss: {', '.join([f'{lv:.4e}' for lv in loss_values])}, "
                f"Each weight: {', '.join([f'{w:.4e}' for w in weights])}, "
                f"Current Re range: ({cur_re_range[0]:.2f}, {cur_re_range[1]:.2f})"
            )
            writer.add_scalar("loss/total", total_loss.item(), epoch)
            writer.add_scalar("info/current_reynolds_max", float(cur_re_range[1]), epoch)
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