import numpy as np

# toy quadratic problem for testing optimization algorithms

def make_quadratic(n=50, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M.T @ M + 1e-1 * np.eye(n)   # make A strictly symmetric positive definite (SPD)
    b = rng.normal(size=n)

    def f(x):  return 0.5 * x @ (A @ x) - b @ x
    def grad(x): return A @ x - b
    def hv(x, v): return A @ v        # constant Hessian
    def xbar(): return np.zeros(n)

    return f, grad, hv, xbar, A, b