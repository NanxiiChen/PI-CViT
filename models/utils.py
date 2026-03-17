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