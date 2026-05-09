import numpy as np


def squared_gradient(center, points):
    """
    Gradient for squared loss.

    f(x,y) = 1/2 ||x-y||^2

    Gradient:
        grad = x - y

    Parameters
    ----------
    center : ndarray
        Feature matrix
    points : ndarray
        Points in the cluster

    Returns
    -------
    ndarray
    Average gradient

    """
    diff = center - points
    return np.mean(diff, axis=0)


def huber_gradient(center, points, delta=1.0):
    """
    Gradient for Huber loss.
    Parameters
    ----------
    center : ndarray
        Feature matrix
    points : ndarray
        Points in the cluster
    delta : float
        Huber threshold

    Returns
    -------
    ndarray
        Average gradient
    """
    diff = center - points
    norm = np.linalg.norm(diff, axis=1, keepdims=True)

    grad = np.where(
        norm <= delta,
        diff,
        delta * diff / (norm + 1e-12)
    )

    return np.mean(grad, axis=0)