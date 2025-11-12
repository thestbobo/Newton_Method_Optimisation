import numpy as np
from linesearch.backtracking import armijo_backtracking



def solve_gradient(cfg, f, grad, x0, max_iters=1000, tol=1e-6):
    x = x0.copy()

    for k in range(max_iters):
        g_x = grad(x)
        g_norm = np.linalg.norm(g_x)
        print(g_norm, tol)
        if g_norm < tol:
            return f"Converged to: x={x} - num_iters={k}"

        alpha = armijo_backtracking(f, x, f(x), g_x, -g_x,
                                    init_alpha=cfg['line_search']['alpha'],
                                    rho=cfg['line_search']['rho'],
                                    c=cfg['line_search']['c'],
                                    max_iters=cfg['line_search']['max_iters'])

        if alpha == 0.0:
            return f"Line search failed at iteration {k}, current x: {x}, current gradient norm: {g_norm}"

        x -= alpha * g_x

    return f"Max iterations reached: x={x}, final_grad_norm={g_norm}, num_iters={max_iters}"