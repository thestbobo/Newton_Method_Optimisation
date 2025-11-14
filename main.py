import argparse
import numpy as np
import yaml

from optim.modified_newton import solve_modified_newton
from optim.truncated_newton import solve_truncated_newton
from optim.gradient_baseline import solve_gradient

from problems.quad import make_quadratic


def main():

    # import configs
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    # set config seed
    np.random.seed(cfg["run"]["seed"])

    # define optimisation problem
    problem = cfg["run"]["problem"]

    if problem == "quad":
        f, grad, hv, xbar, A, b = make_quadratic(
            n=cfg["run"]["n"],
            seed=cfg["run"]["seed"]
        )

    x0 = xbar()
    # define method for optimisation
    method = cfg["run"]["method"]

    res = None
    if method == "gd":
        res = solve_gradient(cfg, f, grad, x0, max_iters=cfg["run"]["max_iters"], tol=cfg["run"]["tolerance"])
    elif method == "mn":
        res = solve_modified_newton()
    elif method == "tn":
        res = solve_truncated_newton()
    else:
        raise ValueError("Unknown optimisation method")

    print(res)
    x = res["x"]
    x_opt = np.linalg.solve(A, b)
    print(f"||x - x*|| = : {np.linalg.norm(x - x_opt)}")
    print(f"f(x) - f(x*) = : {f(x) - f(x_opt)}")


if __name__ == '__main__':
    main()