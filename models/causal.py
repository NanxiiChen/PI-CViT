from typing import Tuple

import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt


class CausalWeightor:
    def __init__(self, num_chunks: int, t_range: Tuple[float, float], **kwargs):
        self.num_chunks = num_chunks
        self.t_range = t_range
        self.bins = jnp.linspace(t_range[0], t_range[1], num_chunks + 1)

    def compute_causal_weights(self, loss_chunks: jnp.ndarray, eps: float = 0.1):
        # loss_chunks: (num_chunks,)
        cumulative_loss = jnp.cumsum(loss_chunks[:-1])
        weights = jnp.concatenate(
            [jnp.array([1.0]), jnp.exp(-eps * cumulative_loss)]
        )  # (num_chunks,)
        return jax.lax.stop_gradient(weights)

    def compute_causal_loss(
        self, residuals: jnp.ndarray, ts: jnp.ndarray, eps: float = 0.1
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # residuals: (B, N_query)
        ts = ts.reshape(-1)  # (N_query,)
        indices = jnp.digitize(ts, self.bins) - 1
        def comute_loss_chunk_per_sample(residual_per_sample):
            sum_residuals_sq = jax.ops.segment_sum(
                residual_per_sample**2, indices, self.num_chunks
            )
            count_per_chunk = jax.ops.segment_sum(
                jnp.ones_like(residual_per_sample), indices, self.num_chunks
            )
            loss_chunks = sum_residuals_sq / (count_per_chunk + 1e-12)
            return loss_chunks # (num_chunks,)
        
        loss_chunks = jax.vmap(comute_loss_chunk_per_sample)(residuals)  # (B, num_chunks)
        loss_chunks = jnp.mean(loss_chunks, axis=0)  # (num_chunks,)

        causal_weights = self.compute_causal_weights(loss_chunks, eps)
        causal_weighted_loss = jnp.sum(causal_weights * loss_chunks)

        return causal_weighted_loss, loss_chunks, causal_weights

    def plot_causal_info(
        self, loss_chunks: jnp.ndarray, causal_weights: jnp.ndarray, eps: float
    ):
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

        ax = axes[0]
        ax.plot(jnp.arange(self.num_chunks), loss_chunks, marker="o")
        ax.set_title("Loss Chunks")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Loss Value")

        ax = axes[1]
        ax.plot(jnp.arange(self.num_chunks), causal_weights, marker="o")
        ax.set_title(f"Causal Weights (eps={eps})")
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Weight Value")

        plt.tight_layout()

        return fig

    def update_causal_eps(
        self,
        causal_weights: jnp.ndarray,
        eps: float,
        max_eps: float = 10.0,
        step_size: float = 5.0,
        min_mean_weight: float = 0.2,
        max_min_weight: float = 0.99,
        **kwargs,
    ) -> float:
        new_eps = eps
        min_weight = causal_weights[-1]
        mean_weight = jnp.mean(causal_weights)
        if min_weight > max_min_weight and eps < max_eps:
            new_eps = eps * step_size

        if mean_weight < min_mean_weight:
            new_eps = eps / step_size

        if not isinstance(new_eps, jnp.ndarray):
            new_eps = jnp.array(new_eps)
        return new_eps
