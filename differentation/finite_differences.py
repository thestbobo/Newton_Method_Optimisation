import numpy as np

"""
    f: funzione di cui calcolare gradiente o hessiana
    x: punto in cui si vuole calcolare il gradiente o l'hessiana
    h: tolleranza/step size
    fd_grad: fuzione per il finite differences gradient
    an_grad: funzione per il gradiente analitico
    forward_backward: -1, 0, 1 se si vuole rispettivamente backward, centred, forward difference
"""

def fd_gradient(f, x, h, forward_backward, relative=False):
    """
    Finite-difference gradient.

    Parameters
    ----------
    f : callable
        Objective function f(x) -> float.
    x : np.ndarray
        Point at which to approximate the gradient.
    h : float
        Base step (typically 10^{-k}).
    forward_backward : int
        0  -> central difference
        +1 -> forward difference
        -1 -> backward difference
    relative : bool
        If True, use component-wise steps h_i = h * |x_i| (fallback to h if x_i=0).
        If False, use constant step h.

    Returns
    -------
    gradf : np.ndarray
        Finite-difference approximation of ∇f(x).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if forward_backward not in (-1, 0, 1):
        raise ValueError("forward_backward must be one of {-1, 0, +1}.")

    n = x.size
    gradf = np.zeros(n, dtype=float)

    def step_i(i):
        hi = h * abs(x[i]) if relative else h
        return hi if hi > 0.0 else h  # safeguard for x[i]==0

    if forward_backward == 0:
        # central differences
        for i in range(n):
            hi = step_i(i)
            xh1 = x.copy(); xh1[i] += hi
            xh2 = x.copy(); xh2[i] -= hi
            gradf[i] = (f(xh1) - f(xh2)) / (2.0 * hi)
        return gradf

    # forward/backward differences
    fx0 = f(x)

    sgn = float(forward_backward)
    for i in range(n):
        hi = step_i(i)
        xh = x.copy()
        xh[i] += sgn * hi
        gradf[i] = (f(xh) - fx0) / (sgn * hi)

    return gradf


def fd_hessian(f, x, h, relative=False):
    """
    Finite-difference Hessian (symmetric).

    Parameters
    ----------
    f : callable
        Objective function f(x) -> float.
    x : np.ndarray
        Point at which to approximate the Hessian.
    h : float
        Base step (typically 10^{-k}).
    relative : bool
        If True, use component-wise steps h_i = h * |x_i| (fallback to h if x_i=0).
        If False, use constant step h.

    Returns
    -------
    hess : np.ndarray
        Finite-difference approximation of ∇²f(x).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if h <= 0:
        raise ValueError("h must be positive.")

    n = x.size
    hess = np.zeros((n, n), dtype=float)
    fx = f(x)


    def step_i(i):
        hi = h * abs(x[i]) if relative else h
        return hi if hi > 0.0 else h  # safeguard for x[i]==0

    for i in range(n):
        hi = step_i(i)

        # diagonal: second derivative w.r.t x_i (central)
        xh1 = x.copy(); xh1[i] += hi
        xh2 = x.copy(); xh2[i] -= hi
        fx1 = f(xh1)
        fx2 = f(xh2)
        hess[i, i] = (fx1 - 2.0 * fx + fx2) / (hi ** 2)

        # off-diagonals: mixed partials (keep your original 4-point forward form, but with hi*hj)
        for j in range(i + 1, n):
            hj = step_i(j)

            x11 = x.copy()
            x11[i] += hi
            x11[j] += hj
            f11 = f(x11)

            xi = x.copy()
            xi[i] += hi
            fi = f(xi)

            xj = x.copy()
            xj[j] += hj
            fj = f(xj)

            hij = hi * hj
            hess_ij = (f11 - fi - fj + fx) / hij

            hess[i, j] = hess_ij
            hess[j, i] = hess_ij

    return hess


def hess_from_grad(grad_fn, x, h, forward_backward=0, relative=False):
    """
    Finite-difference Hessian built from a gradient function.

    Parameters
    ----------
    grad_fn : callable
        Gradient function grad_fn(x) -> np.ndarray.
    x : np.ndarray
        Point at which to approximate the Hessian.
    h : float
        Base step (typically 10^{-k}).
    forward_backward : int
        0  -> central difference
        +1 -> forward difference
        -1 -> backward difference
    relative : bool
        If True, use component-wise steps h_i = h * |x_i| (fallback to h if x_i=0).
        If False, use constant step h.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if forward_backward not in (-1, 0, 1):
        raise ValueError("forward_backward must be one of {-1, 0, +1}.")

    n = x.size
    hess = np.zeros((n, n), dtype=float)

    def step_i(i):
        hi = h * abs(x[i]) if relative else h
        return hi if hi > 0.0 else h  # safeguard for x[i]==0

    if forward_backward == 0:
        for j in range(n):
            hj = step_i(j)
            xh1 = x.copy(); xh1[j] += hj
            xh2 = x.copy(); xh2[j] -= hj
            g1 = grad_fn(xh1)
            g2 = grad_fn(xh2)
            hess[:, j] = (g1 - g2) / (2.0 * hj)
        return hess

    g0 = grad_fn(x)
    sgn = float(forward_backward)
    for j in range(n):
        hj = step_i(j)
        xh = x.copy()
        xh[j] += sgn * hj
        g1 = grad_fn(xh)
        hess[:, j] = (g1 - g0) / (sgn * hj)

    return hess


