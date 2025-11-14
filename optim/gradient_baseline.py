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