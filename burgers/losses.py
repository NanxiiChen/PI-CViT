from typing import Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

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
        2D Burgers' equation PDE residual loss.
        u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
        v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
        
        Args:
        u: C, H, W
        x: N_query, 2
        t: N_query, 1
        """
        Lc = cfg.Lc
        Tc = cfg.Tc
        nu = cfg.nu
        
        
        enc_out = model.encoder(u) # (N_patches, emb_dim)
        
        def sol_single(xi, ti):
            # N_query = 1
            # out_dim = 2
            # xi: (2,)
            # ti: (1,)
            sol = model.decoder(enc_out, xi[None, :], ti[None, :]) # 1, 2
            return sol[0, :] # 2,
        
        def compute_pde_terms(xi, ti):
            # xi: (2,)
            # ti: (1,)
            sol = sol_single(xi, ti) # 2,
            # grad_x: (2, 2) -> [[u_x, u_y], [v_x, v_y]]
            grad_x = jax.jacrev(sol_single, argnums=0)(xi, ti) # 2, 2
            # grad_t: (2, 1) -> [[u_t], [v_t]]
            grad_t = jax.jacrev(sol_single, argnums=1)(xi, ti) # 2, 1
            # hess_x: (2, 2, 2) -> [component, x_dim, x_dim]
            hess_x = jax.hessian(sol_single, argnums=0)(xi, ti) # 2, 2, 2
            
            nabla_u = grad_x[0, :] # (2,) [u_x, u_y]
            nabla_v = grad_x[1, :] # (2,) [v_x, v_y]
            u_t = grad_t[0, 0] # ()
            v_t = grad_t[1, 0] # ()
            laplacian_u = jnp.trace(hess_x[0, :, :]) # ()
            laplacian_v = jnp.trace(hess_x[1, :, :]) # ()
            
            return sol,  u_t, v_t, nabla_u, nabla_v, laplacian_u, laplacian_v
        
        sol, u_t, v_t, nabla_u, nabla_v, laplacian_u, laplacian_v = jax.vmap(
            compute_pde_terms, in_axes=(0,0)
        )(x, t)
        # sol: N_query, 2
        # u_t, v_t: N_query,
        # nabla_u, nabla_v: N_query, 2
        # laplacian_u, laplacian_v: N_query,
        pde_u = (
            u_t / Tc
            + jnp.einsum('Nd,Nd->N', sol, nabla_u) / Lc
            - nu * laplacian_u / (Lc**2)
        )
        pde_v = (
            v_t / Tc
            + jnp.einsum('Nd,Nd->N', sol, nabla_v) / Lc
            - nu * laplacian_v / (Lc**2)
        )
        pde = jnp.stack([pde_u, pde_v], axis=-1) # N_query, 2
        pde = jnp.linalg.norm(pde, ord=2, axis=-1)  # N_query,
        return pde
    
    
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
            
        
        
        
if __name__ == "__main__":
    from .periodic_cvit import PeriodicCViT
    
    key = jax.random.PRNGKey(0)
    model_key, *data_key = jax.random.split(key, 4)
    model = PeriodicCViT(
        key,
        lx=1, ly=2,
        grid_size=(64, 64),
        in_channels=2,
    )
    u = jax.random.normal(data_key[0], (4, 2, 64, 64))
    x = jax.random.uniform(data_key[1], (1000, 2), minval=0.0, maxval=1.0)
    t = jax.random.uniform(data_key[2], (1000, 1), minval=0.0, maxval=1.0)
    
    losses = Losses()
    cfg = Config()
    residuals = jax.vmap(
        losses.residual_pde,
        in_axes=(None, 0, None, None, None)
    )(model, u, x, t, cfg)
    
    print("residuals:", residuals.shape)
    
    