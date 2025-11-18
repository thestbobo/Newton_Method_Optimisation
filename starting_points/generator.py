import numpy as np

def generate_single_starting_point(problem, config):
    """
    Restituisce (x0, start_id) in base al config:
    - start_type = 'xbar'  -> x0 = xbar, start_id = 0
    - start_type = 'random' -> x0 random nel cubo [xbar-1, xbar+1],
                               start_id = random_id (1..5)
    """
    run_cfg = config['run']
    start_type = run_cfg.get('start_type', 'xbar')
    random_id = int(run_cfg.get('random_id', 1))

    xbar = np.asarray(problem.x0, dtype=float)
    seed = run_cfg['seed']

    if start_type == 'xbar':
        # starting point suggerito dal paper
        return xbar, 0

    elif start_type == 'random':
        # vogliamo 5 random diversi ma riproducibili:
        # usa seed base + random_id
        rng = np.random.default_rng(seed + random_id)

        low = xbar - 1.0
        high = xbar + 1.0
        x0 = rng.uniform(low, high)

        # start_id = 1..5, coerente con assignment
        return x0, random_id

    else:
        raise ValueError(f"Unknown start_type '{start_type}' (use 'xbar' or 'random').")
