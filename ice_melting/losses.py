from functools import partial
from typing import Any, Callable, Tuple, Union, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import equinox as eqx

from .configs.train_debug import Config
from models.cvit import CViT


class Losses(eqx.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        # No parameters to initialize for now
    
    @eqx.filter_jit
    def residual_pde(self, 
                    model: Union[eqx.Module, CViT], # mostly CViT here
                    u: jnp.ndarray,
                    x: jnp.ndarray,
                    t: jnp.ndarray,
                    cfg: Config,
                    **kwargs
                    ) -> jnp.ndarray:
        """
        Computes the PDE residual for a single spatio-temporal sample.

        Args:
            model: The neural operator model (CViT) that predicts the phase field.
            u: The parametrized input function of shape (C, H, W).
            x: Spatial coordinate vector of shape (D,).
            t: Time coordinate vector of shape (1,).
            cfg: Configuration object containing physical parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            A scalar jax.numpy.ndarray representing the PDE residual at the given point.
        """
        
        sol = model(u, x, t) # (1,)
        phi = sol[0] # scalar
        phi_fn = lambda x, t: model(u, x, t)[0]
        dphi_dt = jax.grad(phi_fn, argnums=1)(x, t)[0] # scalar
        hess_phi = jax.hessian(phi_fn, argnums=0)(x, t) # (D, D)
        laplacian_phi = jnp.trace(hess_phi) # scalar
        
        F_phi = (phi**2 - 1) **2 / 4.0
        dF_dphi = phi**3 - phi
        
        M_val = cfg.M_val
        lbd = cfg.lbd
        epsilon = cfg.epsilon

        pde = (
            dphi_dt
            - M_val * (laplacian_phi - dF_dphi / epsilon**2)
            +lbd * jnp.sqrt(2 * F_phi) / epsilon
        )
        assert pde.shape == (), f"PDE residual shape incorrect: {pde.shape}"
        return pde
    
    def loss_pde(self, 
                 model: eqx.Module,
                 u: jnp.ndarray,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 cfg: Config,
                 **kwargs
                 ) -> Tuple[jnp.ndarray, dict]:
        """
        Computes the mean squared PDE residual loss over a batch of samples.
        
        Args:
            model: The neural operator model (CViT) that predicts the phase field.
            u: Input function batch of shape (B, C, H, W).
            x: Spatial coordinate array of shape (B, D).
            t: Time coordinate array of shape (B, 1).
            cfg: Configuration object containing physical parameters.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A tuple of (mean_squared_loss, aux_dict).
        """

        residuals = jax.vmap(
            self.residual_pde, 
            in_axes=(None, 0, 0, 0, None)
        )(model, u, x, t, cfg)
        return jnp.mean(jnp.square(residuals)), {} # empty dict for auxiliary info if needed later
    
    
    def residual_ic(self,
                    model: Union[eqx.Module, CViT],
                    u: jnp.ndarray,
                    x: jnp.ndarray,
                    t: jnp.ndarray,
                    ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                    **kwargs
                    ) -> jnp.ndarray:
        
        """
        Computes the initial condition residual for a single spatio-temporal sample.
        
        Args:
            model: The neural operator model (CViT) that predicts the phase field.
            u: The parametrized input function of shape (C, H, W).
            x: Spatial coordinate vector of shape (D,).
            t: Time coordinate vector of shape (1,).
            ic_fn: A callable function that takes spatial coordinates and returns the initial condition value.
    
        Returns:
            A scalar jax.numpy.ndarray representing the initial condition residual at the given point.
        """
        
        sol = model(u, x, t) # (1,)
        phi_pred = sol[0] # scalar
        ref = ic_fn(x) # scalar
        ic_residual = phi_pred - ref
        assert ic_residual.shape == (), f"IC residual shape incorrect: {ic_residual.shape}"
        return ic_residual
    
    def loss_ic(self,
                model: eqx.Module,
                u: jnp.ndarray,
                x: jnp.ndarray,
                t: jnp.ndarray,
                ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                **kwargs
                ) -> jnp.ndarray:
        """Computes the mean squared initial condition residual loss over a batch of samples.
        
        Args:
            model: The neural operator model (CViT) that predicts the phase field.
            u: Input function batch of shape (B, C, H, W).
            x: Spatial coordinate array of shape (B, D).
            t: Time coordinate array of shape (B, 1).
            ic_fn: A callable function that takes spatial coordinates and returns the initial condition value.
            
        Returns:
            A scalar jax.numpy.ndarray representing the mean squared initial condition residual loss.
        """

        residuals = jax.vmap(
            self.residual_ic, 
            in_axes=(None, 0, 0, 0, None)
        )(model, u, x, t, ic_fn)
        return jnp.mean(jnp.square(residuals)), {}
    
    
    def loss_fn(self, 
                model: eqx.Module,
                u: jnp.ndarray,
                x: jnp.ndarray,
                t: jnp.ndarray,
                cfg: Config,
                ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                active_losses: Tuple[str, ...] = ("loss_pde", "loss_ic"),
                ) -> Tuple[Tuple[jnp.ndarray, Tuple[list, jnp.ndarray, dict]], eqx.Module]:
        """Computes the total loss and its gradient using a weighted sum of components.
        
        Args:
            model: The neural operator model (CViT) to differentiate.
            u: Input function batch of shape (B, C, H, W).
            x: Spatial coordinate array of shape (B, D).
            t: Time coordinate array of shape (B, 1).
            cfg: Configuration object containing physical parameters.
            ic_fn: A callable function for initial condition.
            active_losses: A tuple of method names (strings) to include in the loss calculation.
            
        Returns:
            A tuple of ((total_loss, aux_data), total_gradient), where aux_data contains 
            (individual_losses, weights, aux_vars).
        """
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn,  has_aux=True)
            (loss, aux), grad = vg_fn(
                model,
                u=u,
                x=x,
                t=t,
                cfg=cfg,
                ic_fn=ic_fn
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux)
            
        weights = self.grad_norm_weights(grads)
        total_loss = jnp.sum(jnp.array(losses) * weights)
        
        # 使用 jax.tree_map 优雅地合并加权梯度
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
        weights = grad_norms[0] / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)




if __name__ == "__main__":

    from jax import random as jr
    key = jr.PRNGKey(0)
    
    model = CViT(out_dim=1, key=key, grid_size=(112, 112))
    losses = Losses()
    
    k_img, k_x, k_t = jr.split(key, 3)
    u = jr.normal(k_img, (16, 3, 112, 112))

    x_coord = jr.uniform(k_x, (16, 2), minval=0.0, maxval=1.0)  # (B, 2) 空间坐标
    t_coord = jr.uniform(k_t, (16, 1), minval=0.0, maxval=1.0)  # (B, 1) 时间坐标
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    t_coord = t_coord.astype(jnp.float32)
    
    cfg = Config()
    def ic_fn(x):
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    
    total_loss, total_grad = losses.loss_fn(
        model,
        u,
        x_coord,
        t_coord,
        cfg,
        ic_fn,
        active_losses=("loss_pde", "loss_ic")
    )
    
    print("Total loss:", total_loss.shape)
   


