import jax
from .soap import scale_by_soap as scale_by_soap
from .soap import soap as soap

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
    
    elif model_name.lower() == "deeponet":
        from .deeponet import DeepONet
        return DeepONet(
            key,
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )