import numpy as np

def armijo_backtracking(f, x, f_x, g_x, p, init_alpha, rho, c, max_iters=20 ):
    dg = np.dot(g_x, p)
    if dg >= 0:
        raise ValueError("Search direction p is not a descent direction.")
    alpha = init_alpha
    for _ in range(max_iters):
        f_x_new = f(x + alpha * p)
        if np.isfinite(f_x_new) and f_x_new <= f_x + c * alpha * dg:
            return alpha
        alpha *= rho
    return 0.0

