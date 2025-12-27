import jax

jax.config.update("jax_default_matmul_precision", "float32") 
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_h, grid_w, indexing="xy")
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return jnp.expand_dims(pos_embed, 0)


class PatchEmbed(eqx.Module):
    proj: nn.Conv2d
    norm: Optional[nn.LayerNorm]
    patch_size: Tuple[int, int]
    emb_dim: int
    use_norm: bool

    def __init__(
        self,
        key: jr.PRNGKey,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        emb_dim: int = 768,
        use_norm: bool = False,
    ):
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
            key=key
        )
        
        if use_norm:
            self.norm = nn.LayerNorm(emb_dim, eps=1e-5)
        else:
            self.norm = None

    def __call__(self, inputs):
        # inputs: (C, H, W)
        x = self.proj(inputs) # (emb_dim, H', W')
        
        # Flatten: (emb_dim, H', W') -> (emb_dim, N) -> (N, emb_dim)
        x = x.reshape(self.emb_dim, -1).transpose()
        
        if self.use_norm:
            # Change: vmap LayerNorm over sequence dimension
            x = jax.vmap(self.norm)(x)
        return x


class MlpBlock(eqx.Module):
    fc1: nn.Linear
    fc2: nn.Linear
    act: Callable

    def __init__(
        self,
        key: jr.PRNGKey,
        dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None
    ):
        k1, k2 = jr.split(key)
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim, key=k1)
        self.fc2 = nn.Linear(hidden_dim, out_dim, key=k2)
        self.act = jax.nn.gelu

    def __call__(self, inputs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SelfAttnBlock(eqx.Module):
    norm1: nn.LayerNorm
    attn: nn.MultiheadAttention
    norm2: nn.LayerNorm
    mlp: MlpBlock

    def __init__(
        self,
        key: jr.PRNGKey,
        num_heads: int,
        emb_dim: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
    ):
        k1, k2 = jr.split(key)
        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=emb_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(
            key=k2,
            dim=emb_dim,
            out_dim=emb_dim,
            hidden_dim=int(emb_dim * mlp_ratio)
        )

    def __call__(self, inputs):
        # inputs: (seq_len, emb_dim)
        x = jax.vmap(self.norm1)(inputs)
        x = self.attn(x, x, x)
        x = x + inputs

        y = jax.vmap(self.norm2)(x)
        y = jax.vmap(self.mlp)(y)

        return x + y


class CrossAttnBlock(eqx.Module):
    norm_q: nn.LayerNorm
    norm_kv: nn.LayerNorm
    attn: nn.MultiheadAttention
    norm_out: nn.LayerNorm
    mlp: MlpBlock

    def __init__(
        self,
        key: jr.PRNGKey,
        num_heads: int,
        emb_dim: int,
        mlp_ratio: int,
        layer_norm_eps: float = 1e-5,
    ):
        k1, k2 = jr.split(key)
        self.norm_q = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        
        self.attn = nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=emb_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1
        )
        
        self.norm_out = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(
            key=k2,
            dim=emb_dim,
            out_dim=emb_dim,
            hidden_dim=int(emb_dim * mlp_ratio)
        )

    def __call__(self, q_inputs, kv_inputs):
        # q_inputs: (N_query, emb_dim)
        # kv_inputs: (N_kv, emb_dim)
        q = jax.vmap(self.norm_q)(q_inputs)
        kv = jax.vmap(self.norm_kv)(kv_inputs)

        x = self.attn(q, kv, kv)
        x = x + q_inputs
        
        y = jax.vmap(self.norm_out)(x)
        y = jax.vmap(self.mlp)(y)

        return x + y

class Encoder(eqx.Module):
    patch_embed: PatchEmbed
    pos_emb: jnp.ndarray
    blocks: List[SelfAttnBlock]
    norm: nn.LayerNorm
    
    def __init__(
        self,
        key: jr.PRNGKey,
        patch_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        emb_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        in_channels: int = 3,
        layer_norm_eps: float = 1e-5,
        pos_emb_init: Callable = get_2d_sincos_pos_embed,
    ):
        k_patch, *k_blocks = jr.split(key, depth + 1)
        
        self.patch_embed = PatchEmbed(
            key=k_patch,
            patch_size=patch_size,
            in_channels=in_channels,
            emb_dim=emb_dim,
            use_norm=False
        )
        
        # Initialize positional embeddings
        # grid_size is (H, W) of the image
        # patch_size is (ph, pw)
        grid_h = grid_size[0] // patch_size[0]
        grid_w = grid_size[1] // patch_size[1]
        
        # pos_emb_init returns (1, N, D), we store (N, D)
        self.pos_emb = pos_emb_init(emb_dim, (grid_h, grid_w))[0]
        
        self.blocks = [
            SelfAttnBlock(
                key=k,
                num_heads=num_heads,
                emb_dim=emb_dim,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps
            ) for k in k_blocks
        ]
        
        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

    def __call__(self, u):
        # u: (C, H, W)
        u = self.patch_embed(u) # (N, D)
        
        # Add pos embed
        # Ensure shapes match. pos_emb is (N, D)
        u = u + self.pos_emb
        
        for block in self.blocks:
            u = block(u)
            
        # u = self.norm(u)
        u = jax.vmap(self.norm)(u)
        return u


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
        # inputs: (..., in_dim)
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x


class Decoder(eqx.Module):
    fourier_embs_x: FourierEmbs
    fourier_embs_t: FourierEmbs
    proj_x: nn.Linear
    blocks: List[CrossAttnBlock]
    norm: nn.LayerNorm
    mlp: Mlp
    hard_cons_fn: Optional[Callable] = None
    
    def __init__(
        self,
        key: jr.PRNGKey,
        fourier_freq: float,
        dec_depth: int,
        dec_num_heads: int,
        dec_emb_dim: int,
        dec_mlp_act: str,
        mlp_ratio: int,
        out_dim: int,
        num_mlp_layers: int,
        enc_emb_dim: int, # Dimension coming from encoder
        coord_dim: int = 3, # Dimension of coordinates + time
        layer_norm_eps: float = 1e-5,
        hard_cons_fn: Optional[Callable] = None,
    ):
        k_four, k_proj, k_mlp, *k_blocks = jr.split(key, 3 + dec_depth)
        
        k_four_x, k_four_t = jr.split(k_four)
        self.fourier_embs_x = FourierEmbs(
            key=k_four_x,
            embed_scale=fourier_freq,
            embed_dim=dec_emb_dim//2,
            input_dim=coord_dim-1
        )
        self.fourier_embs_t = FourierEmbs(
            key=k_four,
            embed_scale=fourier_freq/10,
            embed_dim=dec_emb_dim//2,
            input_dim=1
        )

        
        self.proj_x = nn.Linear(enc_emb_dim, dec_emb_dim, key=k_proj)
        
        self.blocks = [
            CrossAttnBlock(
                key=k,
                num_heads=dec_num_heads,
                emb_dim=dec_emb_dim,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps
            ) for k in k_blocks
        ]
        
        self.norm = nn.LayerNorm(dec_emb_dim, eps=layer_norm_eps)
        
        self.mlp = Mlp(
            key=k_mlp,
            num_layers=num_mlp_layers,
            hidden_dim=dec_emb_dim,
            out_dim=out_dim,
            in_dim=dec_emb_dim,
            act=dec_mlp_act
        )

        self.hard_cons_fn = hard_cons_fn

    def __call__(self, u, param, x, t):
        # u: (N_patch, enc_emb_dim)
        # param: (3,) --> a, b, theta
        # x: (N_query, spatial_dim)
        # t: (N_query, 1)
        
        # Combine spatial and temporal coords
        # coords = jnp.concatenate([x, t], axis=-1) # (N_query, spatial_dim + 1)
        # queries = self.fourier_embs(coords)  # (N_query, dec_emb_dim)
        x_emb = self.fourier_embs_x(x)  # (N_query, dec_emb_dim//2)
        t_emb = self.fourier_embs_t(t)  # (N_query, dec_emb_dim//2)
        queries = jnp.concatenate([x_emb, t_emb], axis=-1)  # (N_query, dec_emb_dim)
        
        # Project encoder output
        keys_values = jax.vmap(self.proj_x)(u)  # (N_patch, dec_emb_dim)
        
        # Cross attention blocks
        for block in self.blocks:
            queries = block(queries, keys_values)
            
        queries = jax.vmap(self.norm)(queries)
        output = jax.vmap(self.mlp)(queries)  # (N_query, out_dim)

        # apply hard constraint if provided
        if self.hard_cons_fn is not None:
            out0 = self.hard_cons_fn(*param, x[:, 0:1], x[:, 1:2],) # (N_query, out_dim)
            output = output * t + out0  # (N_query, out_dim)

        return output


class CViT(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self,
        key: jr.PRNGKey,
        patch_size: Tuple[int, int] = (16, 16),
        grid_size: Tuple[int, int] = (224, 224), # H, W
        in_channels: int = 3,
        emb_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        fourier_freq: float = 1.0,
        dec_depth: int = 2,
        dec_num_heads: int = 8,
        dec_emb_dim: int = 256,
        dec_mlp_act: str = "gelu",
        num_mlp_layers: int = 1,
        out_dim: int = 2,
        layer_norm_eps: float = 1e-5,
        hard_cons_fn: Optional[Callable] = None,
    ):
        k_enc, k_dec = jr.split(key)
        
        self.encoder = Encoder(
            key=k_enc,
            patch_size=patch_size,
            grid_size=grid_size,
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            in_channels=in_channels,
            layer_norm_eps=layer_norm_eps
        )
        
        self.decoder = Decoder(
            key=k_dec,
            fourier_freq=fourier_freq,
            dec_depth=dec_depth,
            dec_num_heads=dec_num_heads,
            dec_emb_dim=dec_emb_dim,
            dec_mlp_act=dec_mlp_act,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            num_mlp_layers=num_mlp_layers,
            enc_emb_dim=emb_dim,
            coord_dim=3, # (x, y, t)
            layer_norm_eps=layer_norm_eps,
            hard_cons_fn=hard_cons_fn
        )


    def __call__(self, u, param, x, t):
        # u: (C, H, W)
        # param: (3,) --> a, b, theta
        # x: (N_query, spatial_dim)
        # t: (N_query, 1)
        
        enc_out = self.encoder(u) # (N_patch, emb_dim)
        dec_out = self.decoder(enc_out, param, x, t) # (N_query, out_dim)

        return dec_out # (N_query, out_dim)


# TODO: Apply hard constraint to initial conditions, varying with input function `u0`

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    model = CViT(key, in_channels=1, out_dim=2, grid_size=(224, 224))

    # Dummy input
    k_img, k_x, k_t = jr.split(key, 3)
    u = jr.normal(k_img, (16, 1, 224, 224)) # B, C=1, H, W

    x_coord = jr.uniform(k_x, (100, 2), minval=0.0, maxval=1.0)  # (N_query, 2) 空间坐标，每个batch都一样
    t_coord = jr.uniform(k_t, (100, 1), minval=0.0, maxval=1.0)  # (N_query, 1) 时间坐标
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    t_coord = t_coord.astype(jnp.float32)
    
    output = jax.vmap(model, in_axes=(0, None, None))(u, x_coord, t_coord)  # (B, N_query, out_dim)
    print("Output shape:", output.shape)