
from typing import Callable, List, Optional, Tuple
import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr


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

    def __call__(self, q_inputs, kv_inputs,):
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
    mlp: Mlp
    act: Callable
    num_tokens: int
    
    def __init__(
        self,
        key: jr.PRNGKey,
        in_channels: int = 1, # for constant `nu` coefficient
        depth: int = 2,
        num_tokens: int = 4,
        emb_dim: int = 256,
        act: str = "gelu",
    ):
        self.mlp = Mlp(
            key=key,
            num_layers=depth,
            hidden_dim=emb_dim * num_tokens,
            out_dim=emb_dim * num_tokens,
            in_dim=in_channels,
            act=act
        )
        self.act = getattr(jax.nn, act)
        self.num_tokens = num_tokens
        

    def __call__(self, u):
        # u: (1,)
        u = self.mlp(u)  # (emb_dim,)
        u = self.act(u)
        # u = jnp.expand_dims(u, axis=0)  # (1, emb_dim)
        u = u.reshape(self.num_tokens, u.shape[-1] // self.num_tokens)  # (..., num_tokens, emb_dim)
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




class Decoder(eqx.Module):
    fourier_embs_x: FourierEmbs
    proj_x: nn.Linear
    blocks: List[CrossAttnBlock]
    norm: nn.LayerNorm
    mlp: Mlp
    dec_depth: int
    dec_emb_dim: int
    
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
        coord_dim: int = 2,
        layer_norm_eps: float = 1e-5,
    ):
        k_four, k_proj, k_mlp, *k_blocks = jr.split(key, 3 + dec_depth)
        self.dec_depth = dec_depth
        self.dec_emb_dim = dec_emb_dim
        
        self.fourier_embs_x = FourierEmbs(
            key=k_four,
            embed_scale=fourier_freq,
            embed_dim=dec_emb_dim,
            input_dim=coord_dim
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


    def __call__(self, u, x):
        # u: (1, enc_emb_dim)
        # x: (N_query, spatial_dim)
        
        # Combine spatial and temporal coords, 
        queries = self.fourier_embs_x(x)  # (N_query, dec_emb_dim)
        keys_values = jax.vmap(self.proj_x)(u)  # (1, dec_emb_dim)

        # Cross attention blocks
        for i, block in enumerate(self.blocks):
            queries = block(queries, keys_values)
            
        queries = jax.vmap(self.norm)(queries)
        output = jax.vmap(self.mlp)(queries)  # (N_query, out_dim)

        return output


class CViT(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self,
        key: jr.PRNGKey,
        in_channels: int = 1,
        emb_dim: int = 768,
        depth: int = 12,
        num_tokens: int = 4,
        mlp_ratio: int = 4,
        fourier_freq: float = 1.0,
        dec_depth: int = 2,
        dec_num_heads: int = 8,
        dec_emb_dim: int = 256,
        dec_mlp_act: str = "gelu",
        num_mlp_layers: int = 1,
        out_dim: int = 3,
        layer_norm_eps: float = 1e-5,
    ):
        k_enc, k_dec = jr.split(key)
        
        self.encoder = Encoder(
            key=k_enc,
            in_channels=in_channels,
            depth=depth,
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            act="gelu"
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
            coord_dim=2, # (x, y)
            layer_norm_eps=layer_norm_eps,
        )


    def __call__(self, u, x):
        # u: (C, H, W)
        # x: (N_query, spatial_dim)
        # t: (N_query, 1)
        
        enc_out = self.encoder(u) # (N_patch, emb_dim)
        dec_out = self.decoder(enc_out, x) # (N_query, out_dim)

        return dec_out # (N_query, out_dim)


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    model = CViT(key, in_channels=1, out_dim=2)

    # Dummy input
    k_img, k_x, k_t = jr.split(key, 3)
    u = jr.normal(k_img, (16, 1,)) # (B, C) 这里假设输入是16个点的常数场

    x_coord = jr.uniform(k_x, (100, 2), minval=0.0, maxval=1.0)  # (N_query, 2) 空间坐标，每个batch都一样
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    
    output = jax.vmap(model, in_axes=(0, None))(u, x_coord)  # (B, N_query, out_dim)
    print("Output shape:", output.shape)