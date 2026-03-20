import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_loss_log_gaussian(
    y,
    sigma=10,
    ci=0.95,
    n_boot=200,
    eps=1e-12,
    random_state=None,
):
    """
    Smooth noisy loss curves in log-space using Gaussian filtering
    with bootstrap confidence intervals.

    Parameters
    ----------
    y : array-like
        Original loss sequence
    sigma : float
        Gaussian smoothing strength (larger = smoother)
    ci : float
        Confidence interval level
    n_boot : int
        Number of bootstrap samples
    eps : float
        Prevent log(0)
    random_state : int or None

    Returns
    -------
    mean : np.ndarray
        Smoothed curve
    lower : np.ndarray
        Lower CI
    upper : np.ndarray
        Upper CI
    """

    rng = np.random.default_rng(random_state)

    y = np.asarray(y)
    logy = np.log(y + eps)

    # ---- main smoothing ----
    smooth_log = gaussian_filter1d(logy, sigma=sigma)

    # ---- bootstrap CI ----
    boots = []

    n = len(logy)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = logy[idx]

        # sort to maintain temporal structure
        sample = sample[np.argsort(idx)]

        sm = gaussian_filter1d(sample, sigma=sigma)
        boots.append(sm)

    boots = np.array(boots)

    alpha = 1 - ci
    lower_log = np.percentile(boots, 100 * alpha / 2, axis=0)
    upper_log = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)

    mean = np.exp(smooth_log)
    lower = np.exp(lower_log)
    upper = np.exp(upper_log)

    return mean, lower, upper



"""
Utilities for config handling and overrides in dataclass-based configs.
"""

import ast
from warnings import warn
import copy
from dataclasses import field, make_dataclass


def config_to_dict(cfg_obj):
    cfg_cls = cfg_obj if isinstance(cfg_obj, type) else type(cfg_obj)
    out = {}
    for k, v in vars(cfg_cls).items():
        if k.startswith("_"):
            continue
        if callable(v) or isinstance(v, property):
            continue
        out[k] = copy.deepcopy(v)
    return out


def auto_cast(s):
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def set_by_path(d, path, value, strict=True):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur:
            if strict:
                raise KeyError(f"Unknown key path: {path}")
            cur[k] = {}
            warn(f"Creating intermediate dict for missing key: {k} in {path}")
        if not isinstance(cur[k], dict):
            raise TypeError(f"Intermediate key is not dict: {k} in {path}")
        cur = cur[k]
    last = keys[-1]
    if strict and last not in cur:
        raise KeyError(f"Unknown key: {path}")
    cur[last] = value


def _as_dataclass_instance(base_cls, values: dict):
    fields = []
    for k, v in values.items():
        if isinstance(v, (dict, list, set)):
            # mutable defaults need default_factory
            fields.append(
                (k, object, field(default_factory=lambda vv=copy.deepcopy(v): copy.deepcopy(vv)))
            )
        else:
            fields.append((k, object, field(default=copy.deepcopy(v))))

    OverrideConfig = make_dataclass(
        cls_name=f"{base_cls.__name__}Override",
        fields=fields,
        bases=(base_cls,),
        frozen=True,
        eq=False,  # keep object hash; avoids unhashable-field issues in JIT static args
    )
    return OverrideConfig()


def apply_overrides(cfg_obj, override_items, strict=True):
    d = config_to_dict(cfg_obj)
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        set_by_path(d, k, auto_cast(v), strict=strict)

    base_cls = cfg_obj if isinstance(cfg_obj, type) else type(cfg_obj)
    return _as_dataclass_instance(base_cls, d)