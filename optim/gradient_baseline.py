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
        
        
                
def conjugate_gradient_hess_vect_prod(x0, grad_x0, max_iter, eta, hess_vect_prod, hv_h, f, hv_fb, fd_grad):
    
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

            
        
             
    