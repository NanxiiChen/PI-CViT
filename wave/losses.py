from typing import Tuple, Union, Dict

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
        
        
    @eqx.filter_jit
    def residual_pde(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        2D Wave equation residual
        u_tt = c(x,y)^2 * (u_xx + u_yy)
        
        Args:
        u: C, H, W; C=1: u_fields
        x: N_query, 2
        t: N_query, 1
        """
        Lc = cfg.Lc
        Tc = cfg.Tc
        Nx = cfg.Nx
        Ny = cfg.Ny
        lx = cfg.lx
        ly = cfg.ly
        
        enc_out = model.encoder(u) # (N_patches, emb_dim)
        
        def sol_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :])  # (1, 1)
            return sol[0, 0] # ()
        
        def compute_pde_terms(xi, ti):
            hess_x = jax.hessian(sol_single, argnums=0)(xi, ti)  # (2, 2)
            hess_t = jax.hessian(sol_single, argnums=1)(xi, ti)  # (1, 1)
            laplacian = jnp.trace(hess_x)  # ()
            u_tt = hess_t[0,0]  # ()
            
            return laplacian, u_tt
        
        laplacian, u_tt = jax.vmap(
            compute_pde_terms,
            in_axes=(0, 0)
        )(x, t)  # (N_query,), (N_query,), (N_query,)
        
        c = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x[:, 0:1] / lx) \
            * jnp.sin(2 * jnp.pi * x[:, 1:2] / ly)  # (N_query, 1)
        
        pde = u_tt / (Tc ** 2) - (c[:, 0] ** 2) * laplacian / (Lc ** 2)  # (N_query,)
        
        return pde / 100.0  # scale down for numerical stability
    
    
    def loss_pde(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the PDE residual loss.
        Args:
        u: B, C, H, W
        x: N_query, 2
        t: N_query, 1
        """
        
        residuals = jax.vmap(
            self.residual_pde,
            in_axes=(None, 0, None, None, None)
        )(model, u, x, t, cfg)  # B, N_query
        
        if not cfg.use_causality:
            mse_loss = jnp.mean(jnp.square(residuals))
            return mse_loss, {}
        else:
            residuals_mean, loss_chunks, causal_weights =\
                self.causal_weightor.compute_causal_loss(
                    residuals, t, kwargs.get("causal_eps") 
                )
            return residuals_mean, {
                "loss_chunks": loss_chunks,
                "causal_weights": causal_weights
            }
            
    
    def loss_ic(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        # u: B, C, H, W
        lx = getattr(cfg, 'lx', 1.0)
        ly = getattr(cfg, 'ly', 1.0)
        B, C, H, W = u.shape
        x1 = jnp.linspace(0, lx, W, endpoint=False) # W, 
        x2 = jnp.linspace(0, ly, H, endpoint=False) # H,
        xx, yy = jnp.meshgrid(x1, x2, indexing='xy') # (H, W)
        x_ic = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2)
        t_ic = jnp.zeros((x_ic.shape[0], 1))  # (H*W, 1)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(u, x_ic, t_ic)  # B, (H*W), 2
        
        sol_ref = rearrange(u, 'B C H W -> B (H W) C')  # B, (H*W), 2
        mse_loss = jnp.mean(jnp.square(sol - sol_ref))
        return mse_loss, {}
    
    
    def residual_ic_ut(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        # u: C, H, W
        # x: N_query, 2
        # t: N_query, 1
        
        enc_out = model.encoder(u) # (N_patches, emb_dim)
        
        def sol_single(xi, ti):
            sol = model.decoder(enc_out, xi[None, :], ti[None, :])  # (1, 1)
            return sol[0, 0] # ()
        
        def compute_ut(xi, ti):
            grad = jax.grad(sol_single, argnums=1)(xi, ti)  # (1,)
            return grad[0]  # ()
        
        u_t = jax.vmap(
            compute_ut,
            in_axes=(0, 0)
        )(x, t)  # (N_query,)
        
        return u_t
        
    
    def loss_ic_ut(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        """
        du_dt at t=0 should be zero
        """
        lx = getattr(cfg, 'lx', 1.0)
        ly = getattr(cfg, 'ly', 1.0)
        B, C, H, W = u.shape
        u_t = jax.vmap(
            self.residual_ic_ut,
            in_axes=(None, 0, None, None, None)
        )(model, u, x, t, cfg)  # B, (H*W)
        
        mse_loss = jnp.mean(jnp.square(u_t))
        return mse_loss, {}
        
    def loss_fn(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        coords: Dict[str, jnp.ndarray],
        cfg: Config,
        last_weights: jnp.ndarray,
        alpha_w: float = 1.0,
        weight_coef: jnp.ndarray = jnp.array([1.0, 1.0]),
        active_losses: Tuple[str] = ("loss_pde", "loss_ic", "loss_ic_ut"),
        **kwargs
    ) -> Tuple[Tuple[jnp.ndarray, Tuple[list, jnp.ndarray, dict]], eqx.Module]:
    
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            # coord_samples = coords.get(name.split('_')[-1], None)
            coord_samples = coords.get(re.sub(r'loss_', '', name), None)
            if coord_samples is not None:
                x = coord_samples[:, :-1]
                t = coord_samples[:, -1:]
            else:
                x = None
                t = None
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn, has_aux=True)
            (loss, aux), grad = vg_fn(
                model,
                u=u,
                x=x, # for ic, coords will be generated inside the loss function, this arg is ignored
                t=t,  # for ic, this arg is ignored
                cfg=cfg,
                **kwargs
            )
            grad = jax.tree.map(lambda g: jnp.nan_to_num(g), grad)
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux)
            
        if getattr(cfg, "use_gradnorm", True):
            weights = self.grad_norm_weights(grads)
        else:
            weights = jnp.ones(len(active_losses))
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
            return jnp.linalg.norm(r)

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / grad_norms
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)


        
        
        
        
if __name__ == "__main__":
    from .periodic_cvit import PeriodicCViT
    
    key = jax.random.PRNGKey(0)
    model_key, *data_key = jax.random.split(key, 4)
    model = PeriodicCViT(
        key,
        lx=1, ly=1,
        grid_size=(64, 64),
        in_channels=1,
        out_dim=1
    )
    u = jax.random.normal(data_key[0], (4, 1, 64, 64))
    x = jax.random.uniform(data_key[1], (1000, 2), minval=0.0, maxval=1.0)
    t = jax.random.uniform(data_key[2], (1000, 1), minval=0.0, maxval=1.0)
    
    losses = Losses()
    cfg = Config()
    residuals = jax.vmap(
        losses.residual_pde,
        in_axes=(None, 0, None, None, None)
    )(model, u, x, t, cfg)
    
    print("residuals:", residuals.shape)
    
    ic_loss = losses.loss_ic(model, u, cfg)[0]
    print("ic_loss:", ic_loss)
    
    ic_ut_loss = losses.loss_ic_ut(model, u, cfg)[0]
    print("ic_ut_loss:", ic_ut_loss)
    
    