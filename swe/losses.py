from typing import Callable, Tuple, Union, Dict
import re

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from einops import rearrange

from models.cvit import CViT
from models.deeponet import DeepONet
from models.causal import CausalWeightor
from .configs.train_cvit import Config



class Losses(eqx.Module):
    causal_weightor: CausalWeightor = None
    
    def __init__(self, causal_weightor: CausalWeightor = None,
                 *args, **kwargs):
        super().__init__()
        # No parameters to initialize for now
        self.causal_weightor = causal_weightor
        
        
    def residual_continuity(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        2D shallow water equations - continuity equation residual
        h_t + H*(u_x + v_y) = 0
        
        Args:
        u: (C, H, W)
        x: (N_query, 2)
        t: (N_query, 1)
        """
        Lc = cfg.Lc
        Tc = cfg.Tc
        H = cfg.H_val
        
        enc_out = model.encoder(u) # (N_patches, emb_dim)
        
        def sol_u_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :]) # (1, 3) --> h, u, v
            return sol[0, 1:] # (u, v)
        
        def sol_h_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :]) # (1, 3) --> h, u, v
            return sol[0, 0] # h
        
        def compute_pde_terms(xi, ti):
            # nabla_u: (2, 2) --> [[du/dx, du/dy],
            #                      [dv/dx, dv/dy]]
            nabla_u = jax.jacfwd(sol_u_single, argnums=0)(xi, ti) # (2, 2)
            div_v = jnp.trace(nabla_u)  # du/dx + dv/dy, ()
            h_t = jax.grad(sol_h_single, argnums=1)(xi, ti)  # (1，)
            h_t = jnp.squeeze(h_t, axis=-1)  # ()
            return h_t, div_v
        
        h_t, div_v = jax.vmap(compute_pde_terms, 
                              in_axes=(0, 0))(x, t)  # (N_query,), (N_query,)
        
        pde = h_t / Tc + H * div_v / Lc  # (N_query,)
        return pde
    
    def loss_continuity(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        MSE loss for continuity equation residual
        
        Args:
        u: (B, C, H, W)
        x: (N_query, 2)
        t: (N_query, 1)
        """
        
        residuals = jax.vmap(
            self.residual_continuity,
            in_axes=(None, 0, None, None, None)
        )(model, u, x, t, cfg)  # (B, N_query
        
        if not cfg.use_causality:
            mse_loss = jnp.mean(jnp.square(residuals))
            return mse_loss, {}
        
        else:
            residuals_mean, loss_chunks, causal_weights =\
                self.causal_weightor.compute_causal_loss(
                    residuals, t, kwargs.get("causal_eps", 1e-3) 
                )
                
            return residuals_mean, {
                "loss_chunks_continuity": loss_chunks,
                "causal_weights_continuity": causal_weights
            }
            
    def residual_momentum(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        2D shallow water equations - momentum equation residuals
        u_t - f*v + g*h_x = 0
        v_t + f*u + g*h_y = 0
        
        Args:
        u: (C, H, W)
        x: (N_query, 2)
        t: (N_query, 1)
        """
        
        f = cfg.f_val
        g = cfg.g_val
        Lc = cfg.Lc
        Tc = cfg.Tc
        
        enc_out = model.encoder(u) # (N_patches, emb_dim)
        
        def sol_u_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :]) # (1, 3) --> h, u, v
            return sol[0, 1:] # (u, v)
        
        def sol_h_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :]) # (1, 3) --> h, u, v
            return sol[0, 0] # h
        
        def compute_pde_terms(xi, ti):
            uv_sol = sol_u_single(xi, ti)  # (2,)
            duv_dt = jax.jacfwd(sol_u_single, argnums=1)(xi, ti)  # (2,1)
            duv_dt = jnp.squeeze(duv_dt, axis=-1)  # (2,)
            nabla_h = jax.jacfwd(sol_h_single, argnums=0)(xi, ti)  # (2,)
            return (uv_sol[0], uv_sol[1],  # u, v
                    duv_dt[0], duv_dt[1],  # u_t, v_t
                    nabla_h[0], nabla_h[1])  # h_x, h_y
        
        u_sol, v_sol, u_t, v_t, h_x, h_y = jax.vmap(
            compute_pde_terms, in_axes=(0, 0)
        )(x, t)  # 6 个 (N_query,) 数组
        
        pde_u = u_t / Tc - f * v_sol + g * h_x / Lc  # (N_query,)
        pde_v = v_t / Tc + f * u_sol + g * h_y / Lc  # (N_query,)
        pde = jnp.concatenate([pde_u[:, None], pde_v[:, None]], axis=-1)  # (N_query, 2)
        return pde
    
    def loss_momentum(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        MSE loss for momentum equation residuals
        
        Args:
        u: (B, C, H, W)
        x: (N_query, 2)
        t: (N_query, 1)
        """
        
        residuals = jax.vmap(
            self.residual_momentum,
            in_axes=(None, 0, None, None, None)
        )(model, u, x, t, cfg)  # (B, N_query, 2)
        
        if not cfg.use_causality:
            mse_loss = jnp.mean(jnp.square(residuals))
            return mse_loss, {}
        
        else:
            residuals = jnp.sqrt(jnp.sum(jnp.square(residuals), axis=-1))  # (B, N_query)
            residuals_mean, loss_chunks, causal_weights =\
                self.causal_weightor.compute_causal_loss(
                    residuals, t, kwargs.get("causal_eps", 1e-3) 
                )
                
            return residuals_mean, {
                "loss_chunks_momentum": loss_chunks,
                "causal_weights_momentum": causal_weights
            }
            
            
    def loss_ic_h(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.array, dict]:
        # u: B, C=1, H, W
        
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        B, C, H, W = u.shape
        x1 = jnp.linspace(0, lx, W, endpoint=False) # periodic bc
        y1 = jnp.linspace(0, ly, H, endpoint=False)
        xx, yy = jnp.meshgrid(x1, y1, indexing="xy")
        x_ic = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2)
        t_ic = jnp.zeros((H*W, 1))  # (H*W, 1)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(u, x_ic, t_ic)  # (B, H*W, C_out)
        h_pred = sol[:, :, 0]  # (B, H*W)
        h_ref = rearrange(u, "B C H W -> B (H W) C")[:, :, 0]  # (B, H*W)
        mse_loss = jnp.mean(jnp.square(h_pred - h_ref))
        return mse_loss, {}

    def loss_ic_uv(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.array, dict]:
        # u: B, C=3, H, W
        
        lx = getattr(cfg, "lx", 1.0)
        ly = getattr(cfg, "ly", 1.0)
        B, C, H, W = u.shape
        x1 = jnp.linspace(0, lx, W, endpoint=False) # periodic bc
        y1 = jnp.linspace(0, ly, H, endpoint=False)
        xx, yy = jnp.meshgrid(x1, y1, indexing="xy")
        x_ic = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2)
        t_ic = jnp.zeros((H*W, 1))  # (H*W, 1)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(u, x_ic, t_ic)  # (B, H*W, C_out)
        uv_pred = sol[:, :, 1:]  # (B, H*W, 2)
        # uv_ref is zero
        mse_loss = jnp.mean(jnp.square(uv_pred))
        return mse_loss, {}
    
    def loss_fn(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        coords: Dict[str, jnp.ndarray],
        cfg: Config,
        last_weights: jnp.ndarray,
        alpha_w: float = 1.0,
        weight_coef: jnp.ndarray = jnp.array([1.0, 1.0, 1.0, 1.0]),
        active_losses: Tuple[str] = ("momentum", "continuity", "ic_h", "ic_uv"),
        **kwargs
    ):
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            
            if name == "loss_momentum" or name == "loss_continuity":
                coord_samples = coords.get("pde", None)
                x = coord_samples[:, :-1]
                t = coord_samples[:, -1:]
                
            elif name == "loss_ic_h" or name == "loss_ic_uv":
                x = None
                t = None            
                
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn, has_aux=True)
            (loss, aux), grad = vg_fn(
                model, 
                u=u,
                x=x,  # for ic, coord is generated inside loss function
                t=t,  # for ic, coord is generated inside loss function
                cfg=cfg,
                **kwargs
            )
            grad = jax.tree.map(lambda g: jnp.nan_to_num(g), grad)
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux)
            
        weights = self.grad_norm_weights(grads)
        weights = weights * weight_coef[:len(active_losses)]
        weights = alpha_w * weights + (1 - alpha_w) * last_weights
     
        total_loss = jnp.sum(jnp.array(losses) * weights)
        
        total_grad = jax.tree.map(
            lambda *gs: jnp.sum(jnp.stack([w * g for w, g in zip(weights, gs)]), axis=0),
            *grads
        )

        return (total_loss, (losses, weights, aux_vars)), total_grad
        
        
    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            r, _ = ravel_pytree(pytree)
            r = jnp.nan_to_num(r) 
            return jnp.linalg.norm(r)
        
        
        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.nan_to_num(grad_norms)
        mean_norm = jnp.mean(grad_norms)
        safe_norms = jnp.maximum(grad_norms, 1e-8)
        weights = mean_norm / safe_norms
        weights = jnp.nan_to_num(weights)
        # grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        # weights = jnp.mean(grad_norms) / grad_norms
        # weights = jnp.nan_to_num(weights)
        # weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)

        
if __name__ == "__main__":
    from .periodic_cvit import PeriodicCViT
    
    key = jax.random.PRNGKey(0)
    model_key, *data_key = jax.random.split(key, 4)
    model = PeriodicCViT(
        key,
        lx=1, ly=1,
        grid_size=(32, 32),
        patch_size=(8, 8),
        emb_dim=64,
        depth=4,
        num_heads=4,
        dec_depth=2,
        dec_emb_dim=128,
        dec_num_heads=8,
        in_channels=1,
        out_dim=3,
    )
    u = jax.random.normal(data_key[0], (4, 1, 32, 32))
    x = jax.random.uniform(data_key[1], (1000, 2), minval=0.0, maxval=1.0)
    t = jax.random.uniform(data_key[2], (1000, 1), minval=0.0, maxval=1.0)    

    losses = Losses()
    cfg = Config()
    residuals = jax.vmap(
        losses.residual_momentum,
        in_axes=(None, 0, None, None, None)
    )(model, u, x, t, cfg)

    
    print("residuals:", residuals.shape)
    
    ic_h_loss = losses.loss_ic_h(model, u, cfg)[0]
    print("ic_h_loss:", ic_h_loss)
    
    ic_uv_loss = losses.loss_ic_uv(model, u, cfg)[0]
    print("ic_uv_loss:", ic_uv_loss)
    
            
        
        
        
        
        