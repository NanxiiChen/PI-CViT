import jax

def get_model(model_name, key, in_channels, out_dim, **kwargs):
    """
    Factory function to create model instances based on the model_name and parameters.
    """
    
    if model_name.lower() == "cvit":
        from .cvit import CViT
        return CViT(
            key,
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )