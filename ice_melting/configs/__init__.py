import importlib
from .train_debug import Config

def load_configs(name: str) -> Config:
    return importlib.import_module(f".{name}", package=__name__).Config()
