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
    
    def __init__(self,*args, **kwargs):
        super().__init__()
        
        
    @eqx.filter_jit
    def residual_momentum(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        2D steady-state lid-driven cavity flow momentum equations residuals.
        u\cdot nabla u + nabla p - nu * laplacian u = 0
        # eisenstein notation:
        u_j * d(u_i)/d(x_j) + d(p)/d(x_i) - nu * d^2(u_i)/d(x_j)^2 = 0
        
        Args:
        u: shape (1,) representing the normalised `nu`
        """
        
        nu = u[0] * cfg.nu_scale  # denormalize
        Lc = cfg.Lc
        
        
        enc_out = model.encoder(u) # (1, emb_dim)
        
        def sol_u_single(xi):
            sol = model.decoder(enc_out, xi[None, :])  # (1, 3) --> (u, v, p)
            return sol[0, :2] # (2,)
        
        def sol_p_single(xi):
            sol = model.decoder(enc_out, xi[None, :])  # (1, 3) --> (u, v, p)
            return sol[0, 2] # ()
        
        def compute_pde_terms(xi):
            u_sol = sol_u_single(xi)  # (2,)
            # nabla_u: [[u_x, u_y],
            #            [v_x, v_y]]
            nabla_u = jax.jacrev(sol_u_single, argnums=0)(xi)  # (2, 2)
            # nabla_p: [p_x, p_y]
            nabla_p = jax.grad(sol_p_single, argnums=0)(xi)  # (2,)
            hess_u = jax.hessian(sol_u_single, argnums=0)(xi)  # (2, 2, 2)
            laplacian_u = jnp.trace(hess_u, axis1=1, axis2=2)  # (2,)
            
            return u_sol, nabla_u, nabla_p, laplacian_u
        
        u_sol, nabla_u, nabla_p, laplacian_u = jax.vmap(
            compute_pde_terms,
            in_axes=(0,),
        )(x)  # (N_query, 2), (N_query, 2, 2), (N_query, 2), (N_query, 2)

        # pde: u\cdot nabla u + nabla p - nu * laplacian u = 0
        convective_term = jnp.einsum('nj,nij->ni', u_sol, nabla_u)  # (N_query, 2)
        pde_residual = convective_term / Lc + nabla_p / Lc - nu * laplacian_u / (Lc**2)  # (N_query, 2)
        
        return pde_residual
    
    def loss_momentum(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute the momentum equation residual loss.
        """
        
        residuals = jax.vmap(
            self.residual_momentum,
            in_axes=(None, 0, None, None),
        )(model, u, x, cfg)  # (N_query, 2)
    
        loss = jnp.mean(jnp.square(residuals)) # ()
        
        return loss, {}
    
    def residual_continuity(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        2D steady-state lid-driven cavity flow continuity equation residuals.
        nabla \cdot u = 0
        
        Args:
        u: shape (1,) representing the normalised `nu`
        """
        
        enc_out = model.encoder(u) # (1, emb_dim)
        
        def sol_u_single(xi):
            sol = model.decoder(enc_out, xi[None, :])  # (1, 3) --> (u, v, p)
            return sol[0, :2] # (2,)
        
        
        def compute_pde_terms(xi):
            # nabla_u: [[u_x, u_y],
            #            [v_x, v_y]]
            nabla_u = jax.jacrev(sol_u_single, argnums=0)(xi)  # (2, 2)
            div_u = jnp.trace(nabla_u)  # ()
            
            return div_u
        
        div_u = jax.vmap(
            compute_pde_terms,
            in_axes=(0,),
        )(x)  # (N_query,)
        
        return div_u
    
    
    def loss_continuity(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute the continuity equation residual loss.
        """
        residuals = jax.vmap(
            self.residual_continuity,
            in_axes=(None, 0, None, None),
        )(model, u, x, cfg)  # (N_query,)
        loss = jnp.mean(jnp.square(residuals))  # ()
        
        return loss, {}
    
    def loss_bc_walls(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x_bc: jnp.ndarray,
        cfg: Config,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute the boundary condition loss at the walls (no-slip).
        u = 0 at y=0 and y=ly
        v = 0 at x=0 and x=lx
        """
        
        
        sol = jax.vmap(
            model, in_axes=(0, None)
        )(u, x_bc)  # (N_bc, 3)
        
        loss_u = jnp.mean(jnp.square(sol[:, 0]))  # ()
        loss_v = jnp.mean(jnp.square(sol[:, 1]))  # ()
        
        return loss_u + loss_v, {}
    
    def loss_bc_lid(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x_lid: jnp.ndarray,
        cfg: Config,
    ):
        """
        Compute the boundary condition loss at the lid.
        u = 1 at y=ly
        v = 0 at y=ly
        """
        
        
        sol = jax.vmap(
            model, in_axes=(0, None)
        )(u, x_lid)  # (N_bc, 3)
        
        ref = jnp.array([1.0, 0.0])  # (2,)
        loss = jnp.mean(jnp.square(sol[:, :2] - ref[None, :]))  # ()
        
        return loss, {}
    
    
    def loss_fn(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        coords: Dict[str, jnp.ndarray],
        cfg: Config,
        last_weights: jnp.ndarray,
        alpha_w: float = 1.0,
        weight_coef: jnp.ndarray = jnp.array([1.0, 1.0, 1.0, 1.0]),
        active_losses: Tuple[str] = ("loss_momentum", "loss_continuity", "loss_bc_walls", "loss_bc_lid"),
        **kwargs
    ):
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            if name == "loss_momentum":
                x = coords.get("pde", None)
                
            elif name == "loss_continuity":
                x = coords.get("pde", None)
                
            elif name == "loss_bc_walls":
                x = coords.get("bc_walls", None)
                
            elif name == "loss_bc_lid":
                x = coords.get("bc_lid", None)
                
            else:
                x = None
                
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn, has_aux=True)
            (loss, aux), grad = vg_fn(
                model, u=u, x=x, cfg=cfg
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
            return jnp.linalg.norm(r)

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / grad_norms
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)


        
if __name__ == "__main__":
    from .simplified_cvit import CViT
    
    key = jax.random.PRNGKey(0)
    model_key, *data_key = jax.random.split(key, 4)
    model = CViT(
        in_channels=1,
        emb_dim=256,
        depth=2,
        dec_depth=2,
        dec_num_heads=4,
        dec_emb_dim=256,
        out_dim=3,
        key=model_key,
    )
    u = jax.random.randint(data_key[0], (16, 1), 0, 2).astype(jnp.float32)  # (16, 1)
    x_pde = jax.random.uniform(data_key[1], (1024, 2), minval=0.0, maxval=1.0)  # (1024, 2)
    losses = Losses()
    cfg = Config()
    residuals = jax.vmap(
        losses.residual_momentum,
        in_axes=(None, 0, None, None),
    )
    print(residuals(model, u, x_pde, cfg).shape)  # (16, 1024, 2)