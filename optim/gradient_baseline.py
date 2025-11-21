import numpy as np

from linesearch.backtracking import armijo_backtracking



def solve_gradient(cfg, f, grad, x0, max_iters=1000, tol=1e-6):
    x = x0.copy()

    for k in range(max_iters):
        g_x = grad(x)
        g_norm = np.linalg.norm(g_x)

        if g_norm < float(tol):
            print("convergence achieved")
            return {"x": x, "g_norm": g_norm, "iters": k}

        alpha = armijo_backtracking(f, x, f(x), g_x, -g_x,
                                    init_alpha=cfg['line_search']['alpha'],
                                    rho=cfg['line_search']['rho'],
                                    c=cfg['line_search']['c'],
                                    max_iters=cfg['line_search']['max_ls_iter'])

        if alpha == 0.0:
            print("Linesearch failed")
            return {"x": x, "g_norm": g_norm, "iters": k}

        x -= alpha * g_x
    print("max iterations reached")
    return {"x": x, "g_norm": g_norm, "iters": k}


def conjugate_gradient(x0, grad_x0, hess_x0, max_iter, eta):
    
    d = -grad_x0
    z = np.zeros(x0.size)
    r = -grad_x0
    B = hess_x0
    C = grad_x0
    
    for i in range(max_iter):
        
        B_d = B@d
         
        if np.transpose(d)@B_d > 0:
            alpha = (r @ r) / (d @ B_d)
            z = z + alpha * d
            new_r = r + alpha * B_d
            beta = np.transpose(new_r)@new_r / np.transpose(r)@r
            r = new_r
            d = r + beta * d
            if np.linalg.norm(B@z - C) < eta*np.linalg.norm(C):
                return z, i
        else:
            return z, i


def conjugate_gradient_hess_vect_prod_old(x0, grad_x0, max_iter, eta, hess_vect_prod, hv_h, f, hv_fb, fd_grad):
    
    C = grad_x0                 # gradiente in x0
    r = -C                      # residuo iniziale
    d = r.copy()                # direzione iniziale
    z = np.zeros_like(x0)       # soluzione iniziale
    
    r_norm0 = np.linalg.norm(C) # per il test inexact
    
    for k in range(max_iter):
        
        # Hessian-vector product
        B_d = hess_vect_prod(f, fd_grad, x0, d, hv_h, hv_fb)
        
        dBd = d @ B_d

        # Curvatura negativa
        if dBd <= 0:
            return z, k
        
        alpha = (r @ r) / dBd
        
        # update solution
        z = z + alpha * d
        
        # update residual
        new_r = r + alpha * B_d
        
        # ----- Inexact Newton stopping criterion -----
        if np.linalg.norm(new_r) <= eta * r_norm0:
            return z, k
        
        beta = (new_r @ new_r) / (r @ r)
        
        d = new_r + beta * d
        r = new_r
    
    return z, max_iter



def conjugate_gradient_hess_vect_prod(grad_x0, Av, max_iter, eta):
    """
    Parameters
    ----------
    grad_x : np.ndarray
        Gradient at the current x (g).
    Av : callable
        Function that, given a direction d, returns H d.
    max_iter : int
        Maximum number of CG iterations.
    eta : float
        Inexact Newton tolerance, we stop when ||r_k|| <= eta * ||r_0||.

    Returns
    -------
    p : np.ndarray
        Approximate solution of H p = -grad_x.
    k : int
        Number of CG iterations performed.
    """
    g = grad_x0
    b = -g

    # initial guess p0 = 0
    p = np.zeros_like(g)

    # residual r0 = b - , p0 = -g
    r = b.copy()
    r_norm0 = np.linalg.norm(r)

    if r_norm0 == 0.0:
        return p, 0

    d = r.copy()

    for k in range(max_iter):
        Ad = Av(d)          # H d
        dAd = d @ Ad

        # negative curvature
        if dAd <= 0:
            return p, k

        alpha = (r @ r) / dAd

        # update solution
        p = p + alpha * d

        # update residual: r_{k+1} = r_k - alpha * A d_k
        new_r = r - alpha * Ad

        # inexact Newton stpping criterion
        if np.linalg.norm(new_r) <= eta * r_norm0:
            return p, k + 1

        beta = (new_r @ new_r) / (r @ r)

        d = new_r + beta * d
        r = new_r

    return p, max_iter
        
             
    