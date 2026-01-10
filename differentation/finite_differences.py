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

# might be useless
def fd_jacobian(f, x, h, forward_backward):
    
    if forward_backward == 0:
        
        fx0 = f(x)
        jac = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = f(xh1)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = f(xh2)
            
            jac[:, j] = (fxh1 - fxh2) / (2*h)
        
        return jac
        
    fx0 = f(x)
    jac = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = f(xh)
        jac[:, j] = (fxh - fx0) / h
        
    return jac


def fd_hess_from_grad(fd_grad, f, x, h, forward_backward):
    
    if forward_backward == 0:
        
        fx0 = fd_grad(f, x, h, forward_backward)
        hess = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = fd_grad(f, xh1, h, forward_backward)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = fd_grad(f, xh2, h, forward_backward)
            
            hess[:, j] = (fxh1 - fxh2) / (2*h)
        
        return hess
        
    fx0 = fd_grad(f, x, h, forward_backward)
    hess = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = fd_grad(f, xh, h, forward_backward)
        hess[:, j] = (fxh - fx0) / h
        
    return hess


def an_hess_from_grad(an_grad, x, h, forward_backward):
    if forward_backward == 0:
        
        fx0 = an_grad(x)
        hess = np.zeros((fx0.size, x.size))
        
        for j in range(0, x.size):
            
            xh1 = np.copy(x)
            xh1[j] += h
            fxh1 = an_grad(xh1)
            
            xh2 = np.copy(x)
            xh2[j] -= h
            fxh2 = an_grad(xh2)
            
            hess[:, j] = (fxh1 - fxh2) / (2*h)
        
        return hess
        
    fx0 = an_grad(x)
    hess = np.zeros((fx0.size, x.size))
    
    h = h * forward_backward
    
    for j in range(0, x.size):
        
        xh = np.copy(x)
        xh[j] += h
        fxh = an_grad(xh)
        hess[:, j] = (fxh - fx0) / h
        
    return hess


def hessvec_fd_from_grad(f, fd_grad, x, v, h, forward_backward):

    grad_x = fd_grad(x)
    
    x_eps = x + h*v
    grad_x_eps = fd_grad(x_eps)
    
    Hv = (grad_x_eps - grad_x) / h
    return Hv



def hessvec_an_from_grad(an_grad, x, v, h):
    
    grad_x = an_grad(x)
    
    x_eps = x + h*v
    grad_x_eps = an_grad(x_eps)
    
    Hv = (grad_x_eps - grad_x) / h
    return Hv
