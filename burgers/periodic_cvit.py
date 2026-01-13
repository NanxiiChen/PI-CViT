import jax
import jax.numpy as jnp
import equinox as eqx

from models.cvit import Encoder, Decoder, CViT


class PeriodicDecoder(eqx.Module):
    original_decoder: Decoder
    lx: float = eqx.field(static=True)
    ly: float = eqx.field(static=True)
    
    def __init__(self, original_decoder: Decoder, 
                 lx: float, ly: float,):
        self.original_decoder = original_decoder
        self.lx = lx
        self.ly = ly
        
    def __call__(self, u, x, t):
        # u: (N_patch, enc_emb_dim)
        # x: (N_query, spatial_dim) -> expected to be (N, 2)
        # t: (N_query, 1)
        
        sinx = jnp.sin(2 * jnp.pi * x[:, 0:1] / self.lx)
        cosx = jnp.cos(2 * jnp.pi * x[:, 0:1] / self.lx)
        siny = jnp.sin(2 * jnp.pi * x[:, 1:2] / self.ly)
        cosy = jnp.cos(2 * jnp.pi * x[:, 1:2] / self.ly)
        
        x_periodic = jnp.concatenate([sinx, cosx, siny, cosy], axis=-1)
        
        return self.original_decoder(u, x_periodic, t)
    
    
class PeriodicCViT(eqx.Module):
    encoder: Encoder
    decoder: PeriodicDecoder
    
    def __init__(
        self, 
        key, 
        lx=1.0, 
        ly=1.0, 
        patch_size=(16, 16),
        grid_size=(224, 224),
        in_channels=3,
        emb_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        fourier_freq=1.0,
        dec_depth=2,
        dec_num_heads=8,
        dec_emb_dim=256,
        dec_mlp_act="gelu",
        num_mlp_layers=1,
        out_dim=2,
        layer_norm_eps=1e-5,
        use_time_film=True,
    ):
        key_enc, key_dec = jax.random.split(key)
        
        self.encoder = Encoder(
            key=key_enc,
            patch_size=patch_size,
            grid_size=grid_size,
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            in_channels=in_channels,
            layer_norm_eps=layer_norm_eps
        )
        
        original_decoder = Decoder(
            key=key_dec,
            fourier_freq=fourier_freq,
            dec_depth=dec_depth,
            dec_num_heads=dec_num_heads,
            dec_emb_dim=dec_emb_dim,
            dec_mlp_act=dec_mlp_act,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            num_mlp_layers=num_mlp_layers,
            enc_emb_dim=emb_dim,
            coord_dim=5, # For periodic spatial features + time dimension
            layer_norm_eps=layer_norm_eps,
            use_time_film=use_time_film
        )
        
        self.decoder = PeriodicDecoder(original_decoder, lx, ly)
        
    def __call__(self, u, x, t):
        enc_out = self.encoder(u) # (N_patch, enc_emb_dim)
        dec_out = self.decoder(enc_out, x, t) # (N_query, output_dim)
        return dec_out
    
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = CViT(key, in_channels=1, out_dim=2, grid_size=(64, 64))

    # Dummy input
    k_img, k_x, k_t = jax.random.split(key, 3)
    u = jax.random.normal(k_img, (8, 1, 64, 64)) # B, C=1, H, W

    x_coord = jax.random.uniform(k_x, (1000, 2), minval=0.0, maxval=1.0)  # (N_query, 2) 空间坐标，每个batch都一样
    t_coord = jax.random.uniform(k_t, (1000, 1), minval=0.0, maxval=1.0)  # (N_query, 1) 时间坐标
    
    u = u.astype(jnp.float32)
    x_coord = x_coord.astype(jnp.float32)
    t_coord = t_coord.astype(jnp.float32)
    
    output = jax.vmap(model, in_axes=(0, None, None))(u, x_coord, t_coord)  # (B, N_query, out_dim)
    print("Output shape:", output.shape)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.tricontourf(x_coord[:, 0], x_coord[:, 1], output[0, :, 0], levels=14, cmap="RdBu_r")
    ax.set_title('Output Dimension 1')
    
    ax = axes[1]
    ax.tricontourf(x_coord[:, 0], x_coord[:, 1], output[0, :, 1], levels=14, cmap="RdBu_r")
    ax.set_title('Output Dimension 2')
    
    fig.tight_layout()
    fig.savefig("tmp/periodic_cvit_output.png")
    
    