from typing import Callable, Tuple, Union

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
            
    def loss_pde_fd(
        self,
        model: CViT | DeepONet,
        u: jnp.ndarray,
        cfg: Config,
        **kwargs
    ):
        # compute PDE residual using finite difference
        Nx = cfg.Nx
        Ny = cfg.Ny
        Nt = cfg.Nt
        lx = cfg.lx
        ly = cfg.ly
        nu = cfg.nu
        Tc = cfg.Tc
        Lc = cfg.Lc
        
        dx = lx / Nx
        dy = ly / Ny
        dt = (cfg.temporal_domain[1] - cfg.temporal_domain[0]) / Nt
        
        x = jnp.linspace(0, lx, Nx, endpoint=False)
        y = jnp.linspace(0, ly, Ny, endpoint=False)
        xx, yy = jnp.meshgrid(x, y, indexing='xy') 
        x_pde = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (Nx*Ny, 2)
        ts = jnp.linspace(cfg.temporal_domain[0], cfg.temporal_domain[1], Nt + 1)

        B = u.shape[0]

        def d_dx(f):
            return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dx)

        def d_dy(f):
            return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dy)

        def lap(f):
            f_xx = (jnp.roll(f, -1, axis=-1) - 2.0 * f + jnp.roll(f, 1, axis=-1)) / (dx * dx)
            f_yy = (jnp.roll(f, -1, axis=-2) - 2.0 * f + jnp.roll(f, 1, axis=-2)) / (dy * dy)
            return f_xx + f_yy
        
        # RHS in normalized time tau: d/dtau = Tc * d/dt
        def rhs(state):
            # state: (..., 2, Ny, Nx)
            uu = state[..., 0, :, :]
            vv = state[..., 1, :, :]

            ux = d_dx(uu)
            uy = d_dy(uu)
            vx = d_dx(vv)
            vy = d_dy(vv)

            lap_u = lap(uu)
            lap_v = lap(vv)

            # u_t/Tc + (u·∇u)/Lc - nu*Δu/Lc^2 = 0
            # => du/dtau = -Tc*(u·∇u)/Lc + Tc*nu*Δu/Lc^2
            rhs_u = -Tc * (uu * ux + vv * uy) / Lc + Tc * nu * lap_u / (Lc ** 2)
            rhs_v = -Tc * (uu * vx + vv * vy) / Lc + Tc * nu * lap_v / (Lc ** 2)
            return jnp.stack([rhs_u, rhs_v], axis=-3)  # (..., 2, Ny, Nx)
        
        enc_out = jax.vmap(model.encoder)(u) # (B, N_patches, emb_dim)
        decode_ckpt = jax.checkpoint(
            lambda enc, xq, tq: model.decoder(enc, xq, tq)
        )

        def predict_state(t):
            t_pde = jnp.full((x_pde.shape[0], 1), t)
            sol = jax.vmap(decode_ckpt, in_axes=(0, None, None))(
                enc_out, x_pde, t_pde
            )  # (B, Nx*Ny, 2)
            return rearrange(sol, "b (h w) c -> b c h w", h=int(Ny), w=int(Nx))

        # s0 = predict_state(ts[0])  # (B, 2, Ny, Nx)
        s0 = u # use the given initial condition as s0, which is more accurate than the model prediction at t=0

        def step(carry, t_next):
            s_n, sum_sq = carry
            s_np1 = predict_state(t_next)

            k1 = rhs(s_n)
            k2 = rhs(s_n + 0.5 * dt * k1)
            k3 = rhs(s_n + 0.5 * dt * k2)
            k4 = rhs(s_n + dt * k3)
            s_rk4 = s_n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            defect = s_np1 - s_rk4                      # (B, 2, Ny, Nx)
            defect_sq = jnp.sum(defect * defect, axis=1)  # (B, Ny, Nx), 等价于 norm^2 over channel

            step_sum = jnp.sum(defect_sq)                 # 标量
            step_mean = jnp.mean(defect_sq)               # 标量，对应 residual_t_mean[t]
            return (s_np1, sum_sq + step_sum), step_mean

        (_, total_sq), residual_t_mean = jax.lax.scan(
            step,
            (s0, jnp.array(0.0, dtype=u.dtype)),
            ts[1:]
        )

        mse_loss = total_sq / (B * Nt * Ny * Nx)
        aux = {"residual_t_mean": residual_t_mean}  # (Nt,)

        return mse_loss, aux
        
        
            
    
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
    
    def loss_data(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        sol_ref: jnp.ndarray,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        # data loss on some observed data points (u, x, t)
        sol = jax.vmap(
            model, in_axes=(0, None, None)
        )(u, x, t)  # B, N_query, 2
        mse_loss = jnp.mean(jnp.square(sol - sol_ref))
        return mse_loss, {}
        
        
    def loss_fn(
        self,
        model: Union[DeepONet, CViT],
        u: jnp.ndarray,
        x_pde: jnp.ndarray,
        t_pde: jnp.ndarray,
        cfg: Config,
        last_weights: jnp.ndarray,
        alpha_w: float = 1.0,
        weight_coef: jnp.ndarray = jnp.array([1.0, 1.0]),
        active_losses: Tuple[str] = ("loss_pde", "loss_ic"),
        **kwargs
    ) -> Tuple[Tuple[jnp.ndarray, Tuple[list, jnp.ndarray, dict]], eqx.Module]:
        
        losses = []
        grads = []
        aux_vars = {}
        
        for name in active_losses:
            l_fn = getattr(self, name)
            vg_fn = eqx.filter_value_and_grad(l_fn, has_aux=True)
            if name == "loss_data":
                # for data loss, we need to pass sol_ref in kwargs
                x_data = kwargs["x_data"]  # N_query, 2
                t_data = kwargs["t_data"]  # N_query, 1
                sol_ref = kwargs["sol_ref"]  # B, N_query, 2
                u_data = kwargs["u_data"]  # B, C, H, W
                (loss, aux), grad = vg_fn(
                    model,
                    u=u_data,
                    x=x_data,
                    t=t_data,
                    cfg=cfg,
                    sol_ref=sol_ref,
                )
            elif name == "loss_ic":
                # for ic loss, we only need to pass u
                (loss, aux), grad = vg_fn(
                    model,
                    u=u,
                    x=None, # for ic, this arg is ignored
                    t=None,  # for ic, this arg is ignored
                    cfg=cfg,
                )
            elif name == "loss_pde_fd":
                (loss, aux), grad = vg_fn(
                    model,
                    u=u,
                    x=None, # for pde_fd, this arg is ignored
                    t=None,  # for pde_fd, this arg is ignored
                    cfg=cfg,
                    **kwargs
                )
            else:
                (loss, aux), grad = vg_fn(
                    model,
                    u=u,
                    x=x_pde, 
                    t=t_pde, 
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

        # enlarge ic weight
        # coef = jnp.array([0.5, 10.0])  # pde, ic
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
    
    ic_loss = losses.loss_ic(model, u, cfg)
    print("ic_loss:", ic_loss)

