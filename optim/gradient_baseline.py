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



import numpy as np

def conjugate_gradient_hess_vect_prod(grad_x0, Av, max_iter, eta):
    g = grad_x0
    b = -g

    p = np.zeros_like(g)      # initial guess
    r = b.copy()              # r0 = b - A p0 = b
    r_norm0 = np.linalg.norm(r)

    if r_norm0 == 0.0:
        return p, 0, "converged"

    d = r.copy()
    rr = float(r @ r)

    iters = 0

    for _ in range(max_iter):
        iters += 1
        Ad = Av(d)

        # NaN/inf guard (opzionale ma utile)
        if not np.all(np.isfinite(Ad)):
            return -g.copy(), iters, "breakdown"

        dAd = float(d @ Ad)

        # negative curvature / indefiniteness
        if dAd <= 0.0:
            if iters == 1:
                return -g.copy(), iters, "neg_curv"
            else:
                return p, iters, "neg_curv"   # truncated CG direction (p so far)

        alpha = rr / dAd
        p = p + alpha * d

        r = r - alpha * Ad
        r_norm = np.linalg.norm(r)

        # inexact Newton stopping
        if r_norm <= eta * r_norm0:
            # ensure descent (very important for line search)
            if g @ p >= 0:
                p = -g.copy()
                return p, iters, "fallback_sd"
            return p, iters, "converged"

        rr_new = float(r @ r)
        beta = rr_new / rr
        d = r + beta * d
        rr = rr_new

    # max iters reached
    if g @ p >= 0:
        p = -g.copy()
        return p, iters, "fallback_sd"
    return p, iters, "max_iter"



def pcg_hess_vect_prod(grad_x0, Av, Minv, max_iter, eta):
    g = grad_x0
    b = -g
    p = np.zeros_like(g)

    r = b.copy()
    r_norm0 = np.linalg.norm(r)
    if r_norm0 == 0.0:
        return p, 0

    z = Minv(r)
    d = z.copy()
    rz = float(r @ z)

    for it in range(1, max_iter + 1):
        Ad = Av(d)
        dAd = float(d @ Ad)

        # negative curvature
        if dAd <= 0.0:
            # print('Negative curvature')
            if it == 1:
                p = -g.copy()
            if g @ p >= 0:
                p = -g.copy()
            return p, it
        alpha = rz / dAd
        p = p + alpha * d
        r = r - alpha * Ad

        if np.linalg.norm(r) <= eta * r_norm0:
            if g @ p >= 0:
                p = -g.copy()
                #print('fallback')
            return p, it

        z_new = Minv(r)
        rz_new = float(r @ z_new)
        beta = rz_new / rz

        d = z_new + beta * d
        z = z_new
        rz = rz_new

    if g @ p >= 0:
        p = -g.copy()
        #print('fallback')
    return p, max_iter


