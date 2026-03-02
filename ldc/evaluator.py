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
    data = jnp.load(f"{data_dir}/ldc_solutions_Re50-1000_N{Nx}.npz")
    X = data["X"]  # (Ny, Nx)
    Y = data["Y"]  # (Ny, Nx)
    coords = jnp.stack([
        X.reshape(-1),
        Y.reshape(-1)
    ], axis=-1)  # (Ny*Nx, 2)
    target_reynold = data["Re"]
    target_reynold = target_reynold.reshape(-1, 1)  # (B, 1)
    ref_data = data["data"] # B, 2, H, W
    
    target_reynold_normed = cfg.normalize_re(target_reynold)
    sols = jax.vmap(
        model, in_axes=(0, None)
    )(target_reynold_normed, coords)  # (B, Ny*Nx, 3)
    
    B, _, C = sols.shape
    H = Ny
    W = Nx
    sols = rearrange(sols, "B (H W) C -> B C H W", H=int(H), W=int(W))  # (B, 3, H, W)
    sols = sols[:, :2, ...] # (B, 2, H, W)
    if kwargs.get("mask_corners", True):
        eps = 1e-2
        mask = (X > 1 - eps) & (Y > 1 - eps) | (X < eps) & (Y > 1 - eps)
        sols = jnp.where(mask[None, None, :, :], 0.0, sols)
        ref_data = jnp.where(mask[None, None, :, :], 0.0, ref_data)
    # normed on spation points
    l2 = jnp.sqrt(jnp.mean((sols - ref_data)**2, axis=(-1, -2))) / jnp.sqrt(jnp.mean(ref_data**2, axis=(-1, -2))) 
    total_l2 = jnp.mean(l2) # averaged on batch and channels
        
    fig, axes = plt.subplots(
        nrows=3,
        ncols=B,
        figsize=(1.5 * B, 5),
        subplot_kw={"aspect": "equal",},
    )
    
    for i, this_reynold in enumerate(target_reynold):
        ax = axes[0, i]
        ax.set_axis_off()
        ref_speed = jnp.sqrt(ref_data[i, 0, :, :]**2 + ref_data[i, 1, :, :]**2)
        
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
        speed = jnp.sqrt(sols[i, 0, :, :]**2 + sols[i, 1, :, :]**2)
        ax.contourf(
            X, Y, speed, levels=50, cmap="RdBu_r",
            vmin=0, vmax=1.0
        )
        
        ax = axes[2, i]
        ax.set_axis_off()
        diff = jnp.abs(ref_speed - speed)
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
            
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.1, wspace=0.1
    )
    fig.suptitle(
        f"Rel. L2 Error: {total_l2:.2e}",
        y=1.02,
    )
    
    return fig, total_l2


def evaluate_fno_model(
    model: eqx.Module,
    target_reynold: List[float],
    data_dir: str,
    cfg,
    **kwargs,
):
    """
    FNO版本评估：
    - 输入: Re常数场通道 (B,1,Nx,Ny)
    - 输出: (B,3,Nx,Ny) -> 取前两通道(u,v)
    - 注意与坐标网络不同：这里不传coords
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    data = jnp.load(f"{data_dir}/ldc_solutions_Re50-1000_N{Nx}.npz")
    X = data["X"]  # (Ny, Nx)
    Y = data["Y"]  # (Ny, Nx)
    all_re = data["Re"]  # (N_all,)
    all_ref = data["data"]  # (N_all, 2, Ny, Nx)

    target_reynold = jnp.asarray(target_reynold).reshape(-1)
    idxs = jnp.array([jnp.argmin(jnp.abs(all_re - re)) for re in target_reynold])
    eval_re = all_re[idxs]                  # (B,)
    ref_data = all_ref[idxs]                # (B,2,Ny,Nx)

    # Re -> normalized -> 常数场通道
    re_norm = cfg.normalize_re(eval_re)     # (B,)
    B = re_norm.shape[0]
    u0 = jnp.broadcast_to(re_norm[:, None, None, None], (B, 1, Nx, Ny))  # (B,1,Nx,Ny)

    pred = jax.vmap(model)(u0)  # (B,3,Nx,Ny)
    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError(f"FNO output must be (B,>=2,Nx,Ny), got {pred.shape}")

    sols = pred[:, :2, :, :]               # (B,2,Nx,Ny)
    sols = jnp.transpose(sols, (0, 1, 3, 2))  # -> (B,2,Ny,Nx), 对齐ref/X/Y

    if kwargs.get("mask_corners", True):
        eps = 1e-2
        mask = ((X > 1 - eps) & (Y > 1 - eps)) | ((X < eps) & (Y > 1 - eps))
        sols = jnp.where(mask[None, None, :, :], 0.0, sols)
        ref_data = jnp.where(mask[None, None, :, :], 0.0, ref_data)

    l2 = jnp.sqrt(jnp.mean((sols - ref_data) ** 2, axis=(-1, -2))) / jnp.sqrt(
        jnp.mean(ref_data ** 2, axis=(-1, -2)) + 1e-12
    )
    total_l2 = jnp.mean(l2)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=B,
        figsize=(1.5 * B, 5),
        subplot_kw={"aspect": "equal"},
    )

    for i in range(B):
        ax = axes[0, i]
        ax.set_axis_off()
        ref_speed = jnp.sqrt(ref_data[i, 0] ** 2 + ref_data[i, 1] ** 2)
        ax.contourf(X, Y, ref_speed, levels=50, cmap="RdBu_r", vmin=0, vmax=1.0)
        ax.text(0.5, 1.05, f"Re={int(eval_re[i])}", transform=ax.transAxes, ha="center", va="bottom")

        ax = axes[1, i]
        ax.set_axis_off()
        pred_speed = jnp.sqrt(sols[i, 0] ** 2 + sols[i, 1] ** 2)
        ax.contourf(X, Y, pred_speed, levels=50, cmap="RdBu_r", vmin=0, vmax=1.0)

        ax = axes[2, i]
        ax.set_axis_off()
        diff = jnp.abs(ref_speed - pred_speed)
        diff_cont = ax.contourf(X, Y, diff, levels=50, cmap="viridis")
        fig.colorbar(
            diff_cont,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            orientation="horizontal",
            format="%.3f",
            ticks=jnp.linspace(0, jnp.max(diff), num=4)[1:],
        )

        if i == 0:
            axes[0, i].text(-0.05, 0.5, "Ref. mag.", transform=axes[0, i].transAxes, rotation=90, ha="center", va="center")
            axes[1, i].text(-0.05, 0.5, "Pred. mag.", transform=axes[1, i].transAxes, rotation=90, ha="center", va="center")
            axes[2, i].text(-0.05, 0.5, "Abs. error", transform=axes[2, i].transAxes, rotation=90, ha="center", va="center")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.1, wspace=0.1)
    fig.suptitle(f"Rel. L2 Error: {total_l2:.2e}", y=1.02)
    return fig, total_l2







