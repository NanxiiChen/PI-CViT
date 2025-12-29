from typing import Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from models.cvit import CViT
from models.causal import CausalWeightor
from .configs.train_debug import Config


# TODO: add causality loss
class Losses(eqx.Module):
    causal_weightor: CausalWeightor = None
    
    def __init__(self, causal_weightor: CausalWeightor = None,
                 *args, **kwargs):
        super().__init__()
        # No parameters to initialize for now
        self.causal_weightor = causal_weightor
    
    @eqx.filter_jit
    def residual_pde(self, 
                    model: Union[eqx.Module, CViT], # mostly CViT here
                    u: jnp.ndarray,
                    param: jnp.ndarray,
                    x: jnp.ndarray,
                    t: jnp.ndarray,
                    cfg: Config,
                    **kwargs
                    ) -> jnp.ndarray:
        """
        Computes the PDE residual for a single input function `u` over N_query points.

        Args:
            model: The neural operator model (CViT) that predicts the phase field.
            u: The parametrized input function of shape (C, H, W).
            param: Parameters for input function u, shape (unused here, for API consistency).
            x: Spatial coordinate array of shape (N_query, D).
            t: Time coordinate array of shape (N_query, 1).
            cfg: Configuration object containing physical parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            A vector of shape (N_query,) representing the PDE residuals at the given points.
        """
        
        Lc = cfg.Lc  # Characteristic length scale
        Tc = cfg.Tc  # Characteristic time scale
        
        enc_out = model.encoder(u) # (N_patch, emb_dim)

        def phi_single(xi, ti):
            # N_query = 1
            # out_dim = 1
            sol = model.decoder(enc_out, param, xi[None, :], ti[None, :]) # (1, 1)
            return sol[0, 0] # scalar
        
        def compute_pde_terms(xi, ti):
            phi, (grad_x, grad_t) = jax.value_and_grad(
                phi_single, argnums=(0, 1), has_aux=False
            )(xi, ti)  # scalar, (2,), (1,)

            def grad_x_fn(pos):
                return jax.grad(phi_single, argnums=0)(pos, ti)
            
            basis_x = jnp.array([1.0, 0.0])
            basis_y = jnp.array([0.0, 1.0])

            _, hess_x = jax.jvp(grad_x_fn, (xi,), (basis_x,))
            _, hess_y = jax.jvp(grad_x_fn, (xi,), (basis_y,))

            laplacian_phi = hess_x[0] + hess_y[1]
            return phi, grad_t[0], laplacian_phi
        
        phi, dphi_dt, laplacian_phi = jax.vmap(
            compute_pde_terms, in_axes=(0, 0)
        )(x, t)  # (N_query,), (N_query,), (N_query,)
    

        dphi_dt = dphi_dt / Tc
        laplacian_phi = laplacian_phi / (Lc**2)
        
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
        assert pde.shape == (x.shape[0],), f"PDE residual shape incorrect: {pde.shape}"
        return pde
    
    def loss_pde(self, 
                 model: eqx.Module,
                 u: jnp.ndarray,
                 params: jnp.ndarray,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 cfg: Config,
                 **kwargs
                 ) -> Tuple[jnp.ndarray, dict]:
        """
        Computes the mean squared PDE residual loss over a batch of samples.
        
        Args:
            model: The neural operator model (CViT).
            u: Input function batch of shape (B, C, H, W).
            params: Parameters for input function u, shape (B, ...). But unused here. Just for API consistency.
            x: Shared spatial coordinate array of shape (N_query, D).
            t: Shared time coordinate array of shape (N_query, 1).
            cfg: Configuration object containing physical parameters.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A tuple of (scalar MSE loss, auxiliary information dictionary).
        """

        residuals = jax.vmap(
            self.residual_pde, 
            in_axes=(None, 0, 0,  None, None, None)
        )(model, u, params,  x, t, cfg)  # (B, N_query)
        # assert residuals.shape == (u.shape[0], x.shape[0]), f"Residuals shape incorrect: {residuals.shape}"
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
            
    
    
    def residual_ic(self,
                    model: Union[eqx.Module, CViT],
                    u: jnp.ndarray,
                    param: jnp.ndarray,
                    x: jnp.ndarray,
                    t: jnp.ndarray,
                    ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                    cfg: Config,
                    **kwargs
                    ) -> jnp.ndarray:
        
        """
        Computes the initial condition residual for a single `u` over N_query points.
        
        Args:
            model: The neural operator model (CViT).
            u: The parametrized input function of shape (C, H, W).
            param: Parameters for input function u, which determine the IC. shape (3,)
            x: Spatial coordinate array of shape (N_query, 2).
            t: Time coordinate array of shape (N_query, 1).
            ic_fn: Function mapping spatial coordinates to initial condition values.
                   Args are (a, b, theta, x1, x2, epsilon)
            cfg: Configuration object containing physical parameters.
    
        Returns:
            A vector of shape (N_query,) representing the IC residuals.
        """
        
        sol = model(u, param, x, t) # (N_query, out_dim)
        phi_pred = sol[:, 0] # (N_query,)
        
        
        phi_ref = ic_fn(
            param[0], param[1], param[2], 
            x[:, 0], x[:, 1], cfg.epsilon, 
            Lc=cfg.Lc # pass Lc, so that ic_fn can automatically map to physical coords
        ) # (N_query,)
        ic_residual = phi_pred - phi_ref
        assert ic_residual.shape == (x.shape[0],), f"IC residual shape incorrect: {ic_residual.shape}"
        return ic_residual
    
    def loss_ic(self,
                model: eqx.Module,
                u: jnp.ndarray,
                params: jnp.ndarray,
                x: jnp.ndarray,
                t: jnp.ndarray,
                ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                cfg: Config,
                **kwargs
                ) -> jnp.ndarray:
        """Computes the mean squared initial condition residual loss over a batch of samples.
        
        Args:
            model: The neural operator model (CViT).
            u: Input function batch of shape (B, C, H, W).
            params: params for input function u, which determine the ICs. shape (B, 3)
            x: Shared spatial coordinate array of shape (N_query, D).
            t: Shared time coordinate array of shape (N_query, 1).
            ic_fn: Function mapping spatial coordinates to initial condition values.
            cfg: Configuration object containing physical parameters.
            
        Returns:
            A tuple of (scalar MSE loss, auxiliary information dictionary).
        """

        residuals = jax.vmap(
            self.residual_ic, 
            in_axes=(None, 0, 0, None, None, None, None)
        )(model, u, params, x, t, ic_fn, cfg) # (B, N_query)
        return jnp.mean(jnp.square(residuals)), {}
    


    def residual_irr(self,
                     model: Union[eqx.Module, CViT],
                     u: jnp.ndarray,
                     param: jnp.ndarray,
                     x: jnp.ndarray,
                     t: jnp.ndarray,
                     cfg: Config,
                     **kwargs
                     ) -> jnp.ndarray:
        
        Lc = cfg.Lc  # Characteristic length scale
        Tc = cfg.Tc  # Characteristic time scale
        
        enc_out = model.encoder(u) # (N_patch, emb_dim)

        def phi_single(xi, ti):
            # N_query = 1
            # out_dim = 1
            sol = model.decoder(enc_out, param, xi[None, :], ti[None, :]) # (1, 1)
            return sol[0, 0] # scalar

        dphi_dt = jax.vmap(
            jax.grad(phi_single, argnums=1), in_axes=(0, 0)
        )(x, t)  # (N_query,1)
        dphi_dt = dphi_dt[:, 0]  # (N_query,)

        assert dphi_dt.shape == (x.shape[0],), f"dphi_dt shape incorrect: {dphi_dt.shape}"
        dphi_dt = dphi_dt / Tc

        # enforcing dphi/dt < 0
        residual = jax.nn.relu(dphi_dt)
        return residual

    
    def loss_irr(self,
                 model: eqx.Module,
                 u: jnp.ndarray,
                 params: jnp.ndarray,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 cfg: Config,
                 **kwargs
                 ) -> Tuple[jnp.ndarray, dict]:
        
        residuals = jax.vmap(
            self.residual_irr, 
            in_axes=(None, 0, 0, None, None, None)
        )(model, u, params, x, t, cfg) # (B, N_query)
        return jnp.mean(jnp.square(residuals)), {}
    

        
    
    def loss_fn(self, 
                model: eqx.Module,
                u: jnp.ndarray,
                params: jnp.ndarray,
                pde_coords: jnp.ndarray,
                ic_coords: jnp.ndarray,
                cfg: Config,
                ic_fn: Callable[[jnp.ndarray], jnp.ndarray],
                active_losses: Tuple[str, ...] = ("loss_pde", "loss_ic", "loss_irr"),
                **kwargs,
                ) -> Tuple[Tuple[jnp.ndarray, Tuple[list, jnp.ndarray, dict]], eqx.Module]:
        """Computes the total loss and its gradient using a weighted sum of components.
        
        Args:
            model: The neural operator model (CViT) to differentiate.
            u: Input function batch of shape (B, C, H, W).
            params: params for input function u, which determine the ICs. shape (B, 3).
            pde_coords: Spatial and temporal coordinates for PDE residual evaluation, shape (N_query_pde, D+1).
            ic_coords: Spatial coordinates for IC residual evaluation, shape (N_query_ic, D+1).
            cfg: Configuration object containing physical parameters.
            ic_fn: A callable function for initial condition.
            active_losses: A tuple of method names to include in the loss calculation.
            
        Returns:
            A tuple of ((total_loss, aux_data), total_gradient).
        """
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            if name == "loss_pde":
                x, t = pde_coords[:, :-1], pde_coords[:, -1:]
            elif name == "loss_ic":
                x, t = ic_coords[:, :-1], ic_coords[:, -1:]
            elif name == "loss_irr":
                x, t = pde_coords[:, :-1], pde_coords[:, -1:]
            else:
                raise ValueError(f"Unknown loss component: {name}")
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn,  has_aux=True)
            (loss, aux), grad = vg_fn(
                model,
                u=u,
                params=params,
                x=x,
                t=t,
                cfg=cfg,
                ic_fn=ic_fn,
                **kwargs
            )
            # covnert nan or inf grads to zero
            grad = jax.tree.map(lambda g: jnp.nan_to_num(g), grad)
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux)
            
        weights = self.grad_norm_weights(grads)
        # weights = weights.at[-1].set(2*weights[-1])  # Emphasize IC loss
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
    from jax import random as jr

    from models import get_model
    key = jr.PRNGKey(0)
    
    model = get_model("cvit", key, in_channels=1, out_dim=1, grid_size=(112, 112))
    losses = Losses()

    k_u, k_p, k_x, k_t = jr.split(key, 4)
    u = jr.normal(k_u, (16, 1, 112, 112)) # (B=16, C=1, H=112, W=112) 输入函数
    params = jr.uniform(k_p, (16, 3), minval=0.1, maxval=0.5)  # (B=16, 3) 输入函数参数

    x_coord = jr.uniform(k_x, (100, 2), minval=0.0, maxval=1.0)  # (N_query, 2)
    t_coord = jr.uniform(k_t, (100, 1), minval=0.0, maxval=1.0)  # (N_query, 1)
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    t_coord = t_coord.astype(jnp.float32)
    pde_coords = jnp.concatenate([x_coord, t_coord], axis=-1)  # (100, 3)
    ic_coords = jnp.concatenate([x_coord, jnp.zeros_like(t_coord)], axis=-1)  # IC在t=0

    cfg = Config()
    
    def ic_fn(a, b, theta, x, y, epsilon):
        # a, b, theta: scalars
        x_rot = x * jnp.cos(theta) + y * jnp.sin(theta)
        y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta)

        term = jnp.sqrt((x_rot / a)**2 + (y_rot / b)**2) 
        scale = 2 * a * b / (a + b)
        dist = scale * (1.0 - term)

        u = jnp.tanh(dist / (jnp.sqrt(2) * epsilon)) # shape (H, W) or scalar
        return u
    
    (total_loss, (losses, weights, aux_vars)), total_grad = losses.loss_fn(
        model,
        u,
        params,
        pde_coords,
        ic_coords,
        cfg,
        ic_fn,
        active_losses=("loss_pde", "loss_ic")
    )
    
    print("Total loss:", total_loss)



