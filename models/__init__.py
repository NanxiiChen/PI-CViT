import optax


def get_model(model_name, key, in_channels, out_dim, **kwargs):
    """
    Factory function to create model instances based on the model_name and parameters.
    """

    if model_name.lower() == "cvit":
        from .cvit import CViT

        return CViT(key, in_channels=in_channels, out_dim=out_dim, **kwargs)

    elif model_name.lower() == "deeponet":
        from .deeponet import DeepONet

        return DeepONet(key, in_channels=in_channels, out_dim=out_dim, **kwargs)


def get_optimizer(
    optimizer_name: str,
    init_value: float = 5e-4,
    transition_steps: int = 100,
    decay_rate: float = 0.95,
    staircase: bool = False,
    end_value: float = 1e-6,
    **kwargs,
):
    max_grad_norm = kwargs.get("max_grad_norm", 1.0)
    scheduler = optax.exponential_decay(
        init_value=init_value,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=staircase,
        end_value=end_value,
    )
    if optimizer_name.lower() == "adam":
        if max_grad_norm is None:
            return optax.adam(scheduler)
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(scheduler)
        )

    elif optimizer_name.lower() == "soap":
        from .soap import soap as soap
        soap_kwargs = {
            "b1": kwargs.get("b1", 0.95),
            "b2": kwargs.get("b2", 0.99),
            "precondition_frequency": kwargs.get("precondition_frequency", 5),
            "weight_decay": kwargs.get("weight_decay", 0.0),
            "shampoo_beta": kwargs.get("shampoo_beta", -1),
            "eps": kwargs.get("eps", 1e-8),
            "max_precond_dim": kwargs.get("max_precond_dim", 10000),
        }
        optimizer = soap(learning_rate=scheduler, **soap_kwargs)
        if max_grad_norm is not None:
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optimizer
            )
        return optimizer

    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
