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

rcParams.update(
    {
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
        ],
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
        "axes.prop_cycle": plt.cycler(
            color=[
                "#E64B35",
                "#4DBBD5",
                "#00A087",
                "#3C5488",
                "#F39B7F",
                "#8491B4",
                "#91D1C2",
                "#DC0000",
                "#7E6148",
                "#B09C85",
            ]
        ),
        # "axes.prop_cycle": plt.cycler(color=colors),
    }
)


def evaluate_model(
    model: eqx.Module,
    target_ts: List[float],
    data_dir: str,
    Lc: float,
    Tc: float,
    key: jax.random.PRNGKey,
    **kwargs,
):
    data = jnp.load(f"{data_dir}/swe_solutions.npz")
    x = data["x"]  # (W,)
    y = data["y"]  # (H,)
    fft_times = data["times"]  # (T,)
    ref_sols = data["solutions"]    
    idxs = jnp.array(
        [jnp.argmin(jnp.abs(fft_times - t)) for t in target_ts]
    )
    ref_sols = ref_sols[:, idxs, ...] # B, T, C=3, H, W
    
    B, T, C, H, W = ref_sols.shape
    hs = ref_sols[:, 0, 0:1, ...]  # (B, C=1, H, W)
    
    xx, yy = jnp.meshgrid(x, y, indexing="xy")  # (H, W)
    xx = xx / Lc
    yy = yy / Lc
    x_pde = jnp.stack([
        xx.reshape(-1),
        yy.reshape(-1),
    ], axis=-1)  # (Nx*Ny, 2)
    
    def fwd_fn(carry, t):
        t_pde = jnp.full((x_pde.shape[0], 1), t)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(
            hs, x_pde, t_pde
        ) # B, Ny*Nx, C=3
        return carry, sol
    
    _, sols = jax.lax.scan(
        fwd_fn,
        None,
        jnp.array(target_ts) / Tc,
    ) # sols: T, B, H*W, C=3
    sols = rearrange(
        sols, "t b (h w) c -> b t c h w", h=int(H), w=int(W)
    ) # B, T, C=3, H, W
    
    fig, axes = plt.subplots(
        3,
        len(target_ts),
        figsize=(1.5 * len(target_ts), 5),
        subplot_kw={
            "aspect": "equal",
        },
    )
    batch_key, channel_key = jax.random.split(key)
    batch_th = jax.random.randint(
        batch_key, (), 0, B
    )
    channel_th = jax.random.randint(
        channel_key, (), 0, C
    )
    notation = "h" if channel_th == 0 else "u" if channel_th == 1 else "v"
    for i, tic in enumerate(target_ts):
        ax = axes[0, i]
        ax.set_axis_off()
        ax.contourf(
            xx, yy,
            ref_sols[batch_th, i, channel_th, :, :],
            levels=50,
            cmap="RdBu_r",
        )
        if i == 0:
            ax.text(-0.01, 0.5, f"Ref. {notation}",
                    va="center", ha="right", rotation=90,
                    transform=ax.transAxes)
            
        ax.text(
            0.5, 1.05, rf"$t={tic:.1f}\mathrm{{s}}$",
            va="bottom", ha="center", transform=ax.transAxes,
        )
        
        ax = axes[1, i]
        ax.set_axis_off()
        ax.contourf(
            xx, yy, sols[batch_th, i, channel_th, :, :],
            levels=50, cmap="RdBu_r",
        )
        if i == 0:
            ax.text(-0.01, 0.5, f"Pred. {notation}",
                    va="center", ha="right", rotation=90,
                    transform=ax.transAxes)
            
        ax = axes[2, i]
        ax.set_axis_off()
        diff = jnp.abs(
            ref_sols[batch_th, i, channel_th, :, :] -
            sols[batch_th, i, channel_th, :, :]
        )
        diff_cont = ax.contourf(
            xx, yy, diff, levels=50, cmap="viridis",
        )
        
        if i == 0:
            ax.text(
                -0.01,
                0.5,
                r"Abs. Error $|\phi - \hat{\phi}|$",
                rotation=90,
                va="center",
                ha="right",
                transform=ax.transAxes,
            )
            
        colorbar = fig.colorbar(
            diff_cont, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal",
            format="%.3f", ticks=jnp.linspace(0, jnp.max(diff), num=4)[1:]
        )
        
    total_l2 = jnp.sqrt(jnp.sum((ref_sols - sols) ** 2, axis=(1, 3, 4))) / jnp.sqrt(
        jnp.sum(ref_sols**2, axis=(1, 3, 4))
    )
    total_l2 = jnp.mean(total_l2)
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.1, wspace=0.1
    )
    fig.suptitle(
        f"Exam. {batch_th}, Var. {notation}, Rel. L2 Error: {total_l2:.2e}",
        y=1.02,
    )
    return fig, total_l2
            