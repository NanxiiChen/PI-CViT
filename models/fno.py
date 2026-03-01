from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal


def _grid_2d(nx: int, ny: int, dtype=jnp.float32):
    x = jnp.linspace(0.0, 1.0, nx, dtype=dtype)
    y = jnp.linspace(0.0, 1.0, ny, dtype=dtype)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")  # (nx, ny)
    return jnp.stack([xx, yy], axis=0)  # (2, nx, ny)


def _grid_3d(nt: int, nx: int, ny: int, dtype=jnp.float32):
    t = jnp.linspace(0.0, 1.0, nt, dtype=dtype)
    x = jnp.linspace(0.0, 1.0, nx, dtype=dtype)
    y = jnp.linspace(0.0, 1.0, ny, dtype=dtype)
    tt, xx, yy = jnp.meshgrid(t, x, y, indexing="ij")  # (nt, nx, ny)
    return jnp.stack([tt, xx, yy], axis=0)  # (3, nt, nx, ny)


class SpectralConv2d(eqx.Module):
    real_weights_pos: jnp.ndarray
    imag_weights_pos: jnp.ndarray
    real_weights_neg: jnp.ndarray
    imag_weights_neg: jnp.ndarray
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int

    def __init__(self, in_channels, out_channels, modes_x, modes_y, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        k1, k2, k3, k4 = jax.random.split(key, 4)
        shape = (in_channels, out_channels, modes_x, modes_y)
        self.real_weights_pos = glorot_normal()(k1, shape)
        self.imag_weights_pos = glorot_normal()(k2, shape)
        self.real_weights_neg = glorot_normal()(k3, shape)
        self.imag_weights_neg = glorot_normal()(k4, shape)

    def __call__(self, x):
        # x: (in_channels, nx, ny)
        c, nx, ny = x.shape
        assert c == self.in_channels, f"expected {self.in_channels} channels, got {c}"

        x_rft = jnp.fft.rfftn(x, axes=(-2, -1))  # (c, nx, ny//2+1)
        assert self.modes_x <= nx and self.modes_y <= (
            ny // 2 + 1
        ), "modes_x/modes_y exceed FFT resolution"

        x_pos = x_rft[:, : self.modes_x, : self.modes_y]
        x_neg = x_rft[:, -self.modes_x :, : self.modes_y]

        w_pos = self.real_weights_pos + 1j * self.imag_weights_pos
        w_neg = self.real_weights_neg + 1j * self.imag_weights_neg

        out_pos = jnp.einsum("imn,iomn->omn", x_pos, w_pos)
        out_neg = jnp.einsum("imn,iomn->omn", x_neg, w_neg)

        out_rft = jnp.zeros((self.out_channels, nx, ny // 2 + 1), dtype=x_rft.dtype)
        out_rft = out_rft.at[:, : self.modes_x, : self.modes_y].set(out_pos)
        out_rft = out_rft.at[:, -self.modes_x :, : self.modes_y].set(out_neg)

        return jnp.fft.irfftn(out_rft, s=(nx, ny), axes=(-2, -1))


class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv2d
    activation: Callable = jax.nn.gelu

    def __init__(self, in_channels, out_channels, modes_x, modes_y, activation, key):
        spec_key, bypass_key = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels, modes_x, modes_y, spec_key
        )
        self.bypass_conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), key=bypass_key
        )
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))


class FNO2d(eqx.Module):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    projection: eqx.nn.Conv2d
    add_coords: bool

    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x,
        modes_y,
        width,
        depth,
        activation=jax.nn.gelu,
        key=jax.random.PRNGKey(0),
        add_coords: bool = True,
    ):
        self.add_coords = add_coords
        lift_in = in_channels + (2 if add_coords else 0)

        lifting_key, proj_key, *block_keys = jax.random.split(key, depth + 2)
        self.lifting = eqx.nn.Conv2d(
            lift_in, width, kernel_size=(1, 1), key=lifting_key
        )

        self.fno_blocks = [
            FNOBlock2d(width, width, modes_x, modes_y, activation, block_keys[i])
            for i in range(depth)
        ]

        self.projection = eqx.nn.Conv2d(
            width, out_channels, kernel_size=(1, 1), key=proj_key
        )

    def __call__(self, x):
        # x: (C, nx, ny)
        if self.add_coords:
            _, nx, ny = x.shape
            grid = _grid_2d(nx, ny, dtype=x.dtype)
            x = jnp.concatenate([x, grid], axis=0)

        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        return self.projection(x)


# =========================
# Spatio-temporal one-shot FNO
# input:  (C, nx, ny)
# output: (T, C, nx, ny)
# =========================


class SpectralConv3d(eqx.Module):
    # 4 groups: t/x positive/negative, y is rfft-positive only
    real_pp: jnp.ndarray
    imag_pp: jnp.ndarray
    real_pn: jnp.ndarray
    imag_pn: jnp.ndarray
    real_np: jnp.ndarray
    imag_np: jnp.ndarray
    real_nn: jnp.ndarray
    imag_nn: jnp.ndarray

    in_channels: int
    out_channels: int
    modes_t: int
    modes_x: int
    modes_y: int

    def __init__(self, in_channels, out_channels, modes_t, modes_x, modes_y, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_x = modes_x
        self.modes_y = modes_y

        ks = jax.random.split(key, 8)
        shape = (in_channels, out_channels, modes_t, modes_x, modes_y)
        self.real_pp = glorot_normal()(ks[0], shape)
        self.imag_pp = glorot_normal()(ks[1], shape)
        self.real_pn = glorot_normal()(ks[2], shape)
        self.imag_pn = glorot_normal()(ks[3], shape)
        self.real_np = glorot_normal()(ks[4], shape)
        self.imag_np = glorot_normal()(ks[5], shape)
        self.real_nn = glorot_normal()(ks[6], shape)
        self.imag_nn = glorot_normal()(ks[7], shape)

    def _cmul(self, a, wr, wi):
        w = wr + 1j * wi
        return jnp.einsum("itxy,iotxy->otxy", a, w)

    def __call__(self, x):
        # x: (in_channels, nt, nx, ny)
        c, nt, nx, ny = x.shape
        assert c == self.in_channels, f"expected {self.in_channels} channels, got {c}"

        x_rft = jnp.fft.rfftn(x, axes=(-3, -2, -1))  # (c, nt, nx, ny//2+1)
        assert (
            self.modes_t <= nt and self.modes_x <= nx and self.modes_y <= (ny // 2 + 1)
        ), "modes_t/modes_x/modes_y exceed FFT resolution"

        # 4 corners in (t, x), y only positive low modes
        x_pp = x_rft[:, : self.modes_t, : self.modes_x, : self.modes_y]
        x_pn = x_rft[:, : self.modes_t, -self.modes_x :, : self.modes_y]
        x_np = x_rft[:, -self.modes_t :, : self.modes_x, : self.modes_y]
        x_nn = x_rft[:, -self.modes_t :, -self.modes_x :, : self.modes_y]

        y_pp = self._cmul(x_pp, self.real_pp, self.imag_pp)
        y_pn = self._cmul(x_pn, self.real_pn, self.imag_pn)
        y_np = self._cmul(x_np, self.real_np, self.imag_np)
        y_nn = self._cmul(x_nn, self.real_nn, self.imag_nn)

        out_rft = jnp.zeros((self.out_channels, nt, nx, ny // 2 + 1), dtype=x_rft.dtype)
        out_rft = out_rft.at[:, : self.modes_t, : self.modes_x, : self.modes_y].set(
            y_pp
        )
        out_rft = out_rft.at[:, : self.modes_t, -self.modes_x :, : self.modes_y].set(
            y_pn
        )
        out_rft = out_rft.at[:, -self.modes_t :, : self.modes_x, : self.modes_y].set(
            y_np
        )
        out_rft = out_rft.at[:, -self.modes_t :, -self.modes_x :, : self.modes_y].set(
            y_nn
        )

        return jnp.fft.irfftn(out_rft, s=(nt, nx, ny), axes=(-3, -2, -1))


class FNOBlock3d(eqx.Module):
    spectral_conv: SpectralConv3d
    bypass_conv: eqx.nn.Conv3d
    activation: Callable = jax.nn.gelu

    def __init__(
        self, in_channels, out_channels, modes_t, modes_x, modes_y, activation, key
    ):
        spec_key, bypass_key = jax.random.split(key, 2)
        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, modes_t, modes_x, modes_y, spec_key
        )
        self.bypass_conv = eqx.nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 1, 1), key=bypass_key
        )
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))


class FNO(eqx.Module):
    """
    one-shot spatio-temporal predictor
    input:  u0, shape (C, nx, ny)
    output: u,  shape (C, T, nx, ny)
    """

    lifting: eqx.nn.Conv3d
    fno_blocks: List[FNOBlock3d]
    projection: eqx.nn.Conv3d
    time_steps: int
    add_coords: bool
    padding: tuple[int, int, int] # padding for (t, x, y) 
    
    def __init__(
        self,
        key: jax.random.PRNGKey,
        in_channels: int,
        out_channels: int,
        time_steps: int,
        modes_t: int,
        modes_x: int,
        modes_y: int,
        width: int,
        depth: int,
        activation: Callable | str = jax.nn.gelu,
        add_coords: bool = True,
        padding: tuple[int, int, int] = (0, 0, 0),
    ):
        self.time_steps = time_steps
        self.add_coords = add_coords
        self.padding = padding

        lift_in = in_channels + (3 if add_coords else 0)
        lifting_key, proj_key, *block_keys = jax.random.split(key, depth + 2)

        self.lifting = eqx.nn.Conv3d(
            lift_in, width, kernel_size=(1, 1, 1), key=lifting_key
        )
        act_fn = (
            getattr(jax.nn, activation) if isinstance(activation, str) else activation
        )
        self.fno_blocks = [
            FNOBlock3d(width, width, modes_t, modes_x, modes_y, act_fn, block_keys[i])
            for i in range(depth)
        ]
        self.projection = eqx.nn.Conv3d(
            width, out_channels, kernel_size=(1, 1, 1), key=proj_key
        )

    def __call__(self, u0):
        # u0: (C, nx, ny)
        _, nx, ny = u0.shape
        nt = self.time_steps

        # repeat initial field along time: (C, T, nx, ny)
        x = jnp.repeat(u0[:, None, :, :], nt, axis=1)

        if self.add_coords:
            grid = _grid_3d(nt, nx, ny, dtype=u0.dtype)  # (3, T, nx, ny)
            x = jnp.concatenate([x, grid], axis=0)

        # lift
        x = self.lifting(x)  # (width, T, nx, ny)

        # optional one-sided padding on (t, x, y)
        if any(p > 0 for p in self.padding):
            pad_t, pad_x, pad_y = self.padding
            x = jnp.pad(
                x,
                ((0, 0), (0, pad_t), (0, pad_x), (0, pad_y)),
                mode="constant",
            )

        # FNO trunk
        for blk in self.fno_blocks:
            x = blk(x)

        # project to output channels
        x = self.projection(x)

        # crop back to original (C, T, nx, ny)
        x = x[:, :nt, :nx, :ny]
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 64, 64))  # (C, nx, ny)

    model3d = FNO(
        key=key,
        in_channels=2,
        out_channels=2,
        time_steps=100,
        modes_t=8,
        modes_x=8,
        modes_y=8,
        width=32,
        depth=4,
        add_coords=True,
        padding=(10, 0, 0),
    )
    out3d = model3d(x)
    print(out3d.shape)  # (2, 100, 64, 64)
