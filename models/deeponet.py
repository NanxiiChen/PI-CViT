from typing import Callable, List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn

class Mlp(eqx.Module):
    layers: List[nn.Linear]
    act: Callable

    def __init__(
        self,
        key: jr.PRNGKey,
        num_layers: int,
        hidden_dim: int,
        out_dim: int,
        in_dim: int,
        act: str = "gelu"
    ):
        keys = jr.split(key, num_layers + 1)
        self.layers = []
        
        # Hidden layers
        curr_dim = in_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(curr_dim, hidden_dim, key=keys[i]))
            curr_dim = hidden_dim
            
        # Output layer
        self.layers.append(nn.Linear(curr_dim, out_dim, key=keys[-1]))
        self.act = getattr(jax.nn, act)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class FourierEmbs(eqx.Module):
    kernel: jnp.ndarray
    embed_dim: int

    def __init__(self, key: jr.PRNGKey, embed_scale: float, embed_dim: int, input_dim: int):
        self.embed_dim = embed_dim
        self.kernel = jr.normal(key, (input_dim, embed_dim // 2)) * embed_scale

    def __call__(self, x):
        # x: (..., input_dim)
        proj = jnp.dot(x, self.kernel)
        y = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
        return y
    
    
class Encoder(eqx.Module):
    """
    Original DeepONet Branch Net: Direct MLP on flattened grid.
    """
    conv: nn.Conv2d | None
    mlp: Mlp
    out_dim: int
    basis_dim: int
    use_cnn: bool

    def __init__(
        self,
        key: jr.PRNGKey,
        in_channels: int,
        grid_size: Tuple[int, int],
        # CNN args (kept for compatibility, but may not be used)
        conv_out_channels: int = 32,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        mlp_layers: int = 4,
        mlp_hidden_dim: int = 256,
        basis_dim: int = 256,
        out_dim: int = 1,
        act: str = "tanh",
        use_cnn: bool = False,  # whether to use CNN before MLP
    ):
        self.basis_dim = basis_dim
        self.out_dim = out_dim
        self.use_cnn = use_cnn
        
        if use_cnn:
            # 原来的CNN实现
            k_conv, k_mlp = jr.split(key)
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_kernel_size // 2,  # 保持尺寸
                key=k_conv
            )
            h, w = grid_size
            flatten_dim = conv_out_channels * h * w
        else:
            # 直接展平网格
            self.conv = None
            h, w = grid_size
            flatten_dim = in_channels * h * w
            k_mlp = key
        
        # MLP
        self.mlp = Mlp(
            key=k_mlp,
            num_layers=mlp_layers,
            hidden_dim=mlp_hidden_dim,
            out_dim=basis_dim * out_dim,
            in_dim=flatten_dim,
            act=act
        )

    def __call__(self, u):
        # u: (C, H, W)
        if self.use_cnn:
            x = self.conv(u)  # (C_out, H, W)
            x = jax.nn.gelu(x)  # 添加激活
            x = x.flatten()
        else:
            x = u.flatten()  # 直接展平
        
        coeffs = self.mlp(x)  # (basis_dim * out_dim)
        coeffs = coeffs.reshape(self.out_dim, self.basis_dim)
        return coeffs

class Decoder(eqx.Module):
    """
    Corresponds to the Trunk Net in DeepONet.
    Encodes coordinates (x, t) into basis functions and computes dot product with Branch Net output.
    """
    trunk_mlp: Mlp
    bias: jnp.ndarray
    fourier_embs_x: FourierEmbs
    time_mlp: Mlp

    def __init__(
        self,
        key: jr.PRNGKey,
        coord_dim: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
        basis_dim: int,
        out_dim: int,
        fourier_freq: float,
        emb_dim: int,
        act: str = "gelu"
    ):
        k_trunk, k_four, k_time = jr.split(key, 3)
        
        # Spatial dimension is coord_dim - 1 (assuming last one is time)
        spatial_dim = coord_dim - 1
        
        self.fourier_embs_x = FourierEmbs(
            key=k_four,
            embed_scale=fourier_freq,
            embed_dim=emb_dim,
            input_dim=spatial_dim
        )
        
        self.time_mlp = Mlp(
            key=k_time,
            num_layers=2,
            hidden_dim=emb_dim,
            out_dim=emb_dim * 2, # scale + shift
            in_dim=1,
            act="gelu"
        )

        self.trunk_mlp = Mlp(
            key=k_trunk,
            num_layers=mlp_layers,
            hidden_dim=mlp_hidden_dim,
            out_dim=basis_dim,
            in_dim=emb_dim, # Input is now the embedded features
            act=act
        )
        self.bias = jnp.zeros(out_dim)

    def __call__(self, branch_out, x, t):
        # branch_out: (out_dim, basis_dim) - Coefficients from Encoder
        # x: (N_query, spatial_dim)
        # t: (N_query, 1)
        
        x_emb = self.fourier_embs_x(x) # (N_query, emb_dim)
        t_emb = jax.vmap(self.time_mlp)(t) # (N_query, emb_dim * 2)
        
        scale, shift = jnp.split(t_emb, 2, axis=-1)
        features = x_emb * (1 + scale) + shift # (N_query, emb_dim)
        
        # Evaluate basis functions at coordinates (Trunk Net)
        trunk_out = jax.vmap(self.trunk_mlp)(features) # (N_query, basis_dim)
        
        # DeepONet combination: G(u)(y) = sum_k c_k * phi_k(y) + bias
        # branch_out (C): (out_dim, basis_dim)
        # trunk_out (Phi): (N_query, basis_dim)
        # Result: (N_query, out_dim)
        
        out = jnp.einsum('od, nd -> no', branch_out, trunk_out)
        out = out + self.bias
        
        return out

class DeepONet(eqx.Module):
    encoder: Encoder # Wraps Branch Net
    decoder: Decoder # Wraps Trunk Net + Dot Product

    def __init__(
        self,
        key: jr.PRNGKey,
        # Encoder (Branch) args
        in_channels: int = 3,
        grid_size: Tuple[int, int] = (224, 224),
        branch_use_cnn: bool = False,
        branch_conv_channels: int = 32,
        branch_conv_kernel: int = 4,
        branch_conv_stride: int = 4,
        branch_mlp_layers: int = 2,
        branch_mlp_hidden: int = 128,
        # Decoder (Trunk) args
        trunk_mlp_layers: int = 3,
        trunk_mlp_hidden: int = 128,
        trunk_fourier_freq: float = 1.0,
        trunk_emb_dim: int = 128,
        # Common args
        basis_dim: int = 128,
        out_dim: int = 2,
        coord_dim: int = 3, # x(2) + t(1)
        act: str = "gelu"
    ):
        k_enc, k_dec = jr.split(key)
        
        self.encoder = Encoder(
            key=k_enc,
            in_channels=in_channels,
            grid_size=grid_size,
            use_cnn=branch_use_cnn,
            conv_out_channels=branch_conv_channels,
            conv_kernel_size=branch_conv_kernel,
            conv_stride=branch_conv_stride,
            mlp_layers=branch_mlp_layers,
            mlp_hidden_dim=branch_mlp_hidden,
            basis_dim=basis_dim,
            out_dim=out_dim,
            act=act
        )
        
        self.decoder = Decoder(
            key=k_dec,
            coord_dim=coord_dim,
            mlp_layers=trunk_mlp_layers,
            mlp_hidden_dim=trunk_mlp_hidden,
            basis_dim=basis_dim,
            out_dim=out_dim,
            fourier_freq=trunk_fourier_freq,
            emb_dim=trunk_emb_dim,
            act=act
        )

    def __call__(self, u, x, t):
        # u: (C, H, W)
        # x: (N_query, spatial_dim)
        # t: (N_query, 1)
        
        # Branch Net execution
        coeffs = self.encoder(u) # (out_dim, basis_dim)
        
        # Trunk Net execution + Combination
        out = self.decoder(coeffs, x, t) # (N_query, out_dim)
        
        return out

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    model = DeepONet(key, in_channels=1, out_dim=2, grid_size=(224, 224))

    # Dummy input
    k_img, k_x, k_t = jr.split(key, 3)
    u = jr.normal(k_img, (16, 1, 224, 224)) # B, C=1, H, W

    x_coord = jr.uniform(k_x, (100, 2), minval=0.0, maxval=1.0)  # (N_query, 2)
    t_coord = jr.uniform(k_t, (100, 1), minval=0.0, maxval=1.0)  # (N_query, 1)
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    t_coord = t_coord.astype(jnp.float32)
    
    # vmap over batch dimension (0) for u, but keep coords shared (None) or batched depending on use case
    # Here assuming standard operator learning setup: same coords for one u, or different coords.
    # Following CViT example: vmap over batch of u.
    output = jax.vmap(model, in_axes=(0, None, None))(u, x_coord, t_coord)  # (B, N_query, out_dim)
    print("Output shape:", output.shape)
