import os
from typing import List

from einops import rearrange
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
from .configs.train_cvit import Config

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
    target_reynold: List[float],
    data_dir: str,
    cfg: Config,
    **kwargs,
):
    Nx, Ny = cfg.Nx, cfg.Ny
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="xy") # (Ny, Nx)
    coords = jnp.stack([
        X.reshape(-1),
        Y.reshape(-1)
    ], axis=-1)  # (Ny*Nx, 2)
    target_reynold = jnp.array(target_reynold) # reynolds number
    target_reynold = target_reynold.reshape(-1, 1)  # (B, 1)
    
    target_reynold_normed = cfg.normalize_re(target_reynold)
    sols = jax.vmap(
        model, in_axes=(0, None)
    )(target_reynold_normed, coords)  # (B, Ny*Nx, 3)
    
    B, _, C = sols.shape
    H = Ny
    W = Nx
    sols = rearrange(sols, "B (H W) C -> B C H W", H=int(H), W=int(W))  # (B, 3, H, W)
    u = sols[:, 0, :, :]  # (B, H, W)
    v = sols[:, 1, :, :]  # (B, H, W)
    speed = jnp.sqrt(u**2 + v**2)  # (B, H, W)
    
    fig, axes = plt.subplots(
        nrows=3,
        ncols=B,
        figsize=(1.5 * B, 5),
        subplot_kw={"aspect": "equal",},
    )
    
    total_l2 = 0
    for i, this_reynold in enumerate(target_reynold):
        ax = axes[0, i]
        ax.set_axis_off()
        ref_data = jnp.load(
            f"{data_dir}/flow_fields_Re{int(this_reynold[0])}_N128.npz"
        )
        ref_u = ref_data["u"]  # (H, W)
        ref_v = ref_data["v"]  # (H, W)
        ref_speed = jnp.sqrt(ref_u**2 + ref_v**2)  # (H, W)
        
        ax.contourf(
            X, Y, ref_speed, levels=50, cmap="RdBu_r",
            vmin=0, vmax=1.0
        )
        ax.text(
            0.5, 1.05, f"Re={int(this_reynold[0])}", 
            transform=ax.transAxes,
            ha="center", va="bottom",
        )
        
        ax = axes[1, i]
        ax.set_axis_off()
        ax.contourf(
            X, Y, speed[i], levels=50, cmap="RdBu_r",
            vmin=0, vmax=1.0
        )
        
        ax = axes[2, i]
        ax.set_axis_off()
        diff = jnp.abs(ref_speed - speed[i])
        diff_cont = ax.contourf(X, Y, diff, levels=50, cmap="viridis",)
        colorbar = fig.colorbar(
            diff_cont, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal",
            format="%.3f", ticks=jnp.linspace(0, jnp.max(diff), num=4)[1:]
        ) 
        
        if i == 0:
            axes[0, i].text(
                -0.05, 0.5, "Ref. mag.",
                transform=axes[0, i].transAxes,
                rotation=90, ha="center", va="center",
            )
            axes[1, i].text(
                -0.05, 0.5, "Pred. mag.",
                transform=axes[1, i].transAxes,
                rotation=90, ha="center", va="center",
            )
            axes[2, i].text(
                -0.05, 0.5, "Abs. error",
                transform=axes[2, i].transAxes,
                rotation=90, ha="center", va="center",
            )
            
        this_l2 = jnp.sqrt(jnp.mean(diff**2)) / jnp.sqrt(jnp.mean(ref_speed**2))
        total_l2 += this_l2 / B
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.1, wspace=0.1
    )
    fig.suptitle(
        f"Rel. L2 Error: {total_l2:.2e}",
        y=1.02,
    )
    
    return fig, total_l2
            

        
            
        
        
    
    