import os
from typing import List

from einops import rearrange
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx


from ggsci import pal_npg
colors = pal_npg()(10)
from matplotlib import font_manager
font_dir = "./helvetica/"
font_names = os.listdir(font_dir)
for font_name in font_names:
    font_manager.fontManager.addfont(font_dir + font_name)
# nature style
from matplotlib import rcParams
rcParams.update({
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica",],
    "pdf.fonttype": 42,
    "figure.dpi": 300,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # thin ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    # thin axes
    "axes.linewidth": 0.5,
    # thin legend frame
    "legend.frameon": False,
    # set color_theme as ggsci
    "axes.prop_cycle": plt.cycler(color=[
        "#E64B35", "#4DBBD5", "#00A087",
        "#3C5488", "#F39B7F", "#8491B4",
        "#91D1C2", "#DC0000", "#7E6148", "#B09C85"
    ]),
    # "axes.prop_cycle": plt.cycler(color=colors),
})

def evaluate_model(
    model: eqx.Module,
    target_ts: List[float],
    data_dir: str,
    Lc: float,
    Tc: float,
    key: jax.random.PRNGKey,
    **kwargs
):
    params = jnp.load(f"{data_dir}/initial_params.npy")
    ref_sols = jnp.load(f"{data_dir}/solutions.npy") # (B, fem_steps, C, H, W)
    fem_times = jnp.load(f"{data_dir}/times.npy") # (fem_steps,)
    mesh_grids = jnp.load(f"{data_dir}/mesh_points.npy") # shape: (2, H, W)
    idxs = jnp.array([jnp.argmin(jnp.abs(fem_times - t)) for t in target_ts]) # shape: (T,)
    ref_sols = ref_sols[:, idxs, :, :, :]  # (B, T, C, H, W)
    B, T, C, H, W = ref_sols.shape
    us = ref_sols[:, 0, :, :, :]  # (B, C=1, H, W)
    # us: (B, C=1, H, W)
    # params: (B, P=3) --> (a, b, theta)
    # mesh_grids: (2, H, W)
    # target_ts: List of time steps to evaluate
    # Lc, Tc: characteristic length and time scales
    B, C, H, W = us.shape
    meshx = mesh_grids[0]  # (H, W)
    meshy = mesh_grids[1]  # (H, W)
    x_coord = jnp.stack([meshx.reshape(-1), meshy.reshape(-1)], axis=-1) / Lc  # (H*W, 2)
    
    def fwd_fn(carry, t):
        t_coord = jnp.full(
            (x_coord.shape[0], 1), t
        ) / Tc  # (H*W, 1)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(us, x_coord, t_coord)  # (B, H*W, C=1)
        return carry, sol
    
    _, sols = jax.lax.scan(
        fwd_fn,
        None,
        jnp.array(target_ts),
    )  # sols: (T, B, H*W, C=1)
    # T, B, H*W, C=1 --> B, T, C, H, W
    # sols = sols.transpose(1, 0, 3, 2).reshape(B, len(target_ts), C, H, W)
    sols = rearrange(sols, "t b (h w) c -> b t c h w", h=int(H), w=int(W)) # avoid traced H and W
    fig, axes = plt.subplots(3, len(target_ts), figsize=(1.5 * len(target_ts), 5), subplot_kw={
        "aspect": "equal",
    })
    # batch_th = 0  # only evaluate the first sample in the batch
    # randomly select a batch index to evaluate
    batch_th = jax.random.randint(key, (), 0, B)
    channel_th = 0  # only evaluate the first channel
    for i, tic in enumerate(target_ts):
        # plot reference
        ax = axes[0, i]
        ax.set_axis_off()
        ax.pcolormesh(
            meshx, meshy, 
            ref_sols[batch_th, i, channel_th, :, :], 
            cmap="coolwarm", shading="auto", rasterized=True
        )
        if i == 0:
            ax.text(-0.01, 0.5, r"Ref. $\phi$", 
                    rotation=90, va="center", ha="right", transform=ax.transAxes
            )
        
        ax.text(0.5, 1.05, fr"$t={tic:.1f}\mathrm{{s}}$",
                va="bottom", ha="center", transform=ax.transAxes)
        
        ax = axes[1, i]
        ax.set_axis_off()
        ax.pcolormesh(
            meshx, meshy, 
            sols[batch_th, i, channel_th, :, :], 
            cmap="coolwarm", shading="auto", rasterized=True
        )
        if i == 0:
            ax.text(-0.01, 0.5, r"Pred. $\hat{\phi}$", 
                    rotation=90, va="center", ha="right", transform=ax.transAxes
            )

        ax = axes[2, i]
        ax.set_axis_off()
        diff = jnp.abs(
            ref_sols[batch_th, i, channel_th, :, :] - sols[batch_th, i, channel_th, :, :]
        )
        diff_cont = ax.pcolormesh(
            meshx, meshy,
            diff,
            cmap="coolwarm", shading="auto", rasterized=True
        )
        if i == 0:
            ax.text(-0.01, 0.5, r"Abs. Error $|\phi - \hat{\phi}|$", 
                    rotation=90, va="center", ha="right", transform=ax.transAxes
            )

        colorbar = fig.colorbar(
            diff_cont, ax=ax, fraction=0.046, pad=0.04,
            orientation="horizontal"
        )

    # total_l2 = jnp.linalg.norm(
    #     ref_sols - sols, axis=(1, 3, 4)
    # ) / jnp.linalg.norm(
    #     ref_sols, axis=(1, 3, 4)
    # )  # shape: (B, C)
    total_l2 = jnp.sqrt(
        jnp.sum((ref_sols - sols) ** 2, axis=(1, 3, 4))
    ) / jnp.sqrt(
        jnp.sum(ref_sols ** 2, axis=(1, 3, 4))
    )
    total_l2 = jnp.mean(total_l2)
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.1, wspace=0.1
    )
    return fig, total_l2
        





