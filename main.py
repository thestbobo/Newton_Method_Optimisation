import yaml
import numpy as np

from optim import solve_modified_newton, solve_truncated_newton
from analysis.postprocess import postprocess
from problems import problem_classes   # dict: {'quad': QuadProblem, 'ext_powell': ExtendedPowellProblem, ...}


def main(config):
    """
    Esegue ESATTAMENTE l'esperimento descritto nel config:
    - un problema
    - una dimensione n
    - una modalità derivate (exact / fd_hessian / fd_all)
    - uno o più metodi in run.methods
    - x̄ + num_random_starts punti di partenza

    Ritorna all_results, che poi passa al postprocess.
    """


    # ESTRAZIONE DA CONFIG

    run_cfg = config['run']
    problem_name = run_cfg['problem']      # es. 'quad'
    n = run_cfg['n_value']                 # es. 2, 1000, 10000, 100000
    methods = run_cfg['methods']           # es. ['mn', 'tn']
    seed = run_cfg['seed']
    num_random_starts = run_cfg['num_random_starts']

    # SETTING SEED

    np.random.seed(seed)

    # istanzia problema
    if problem_name not in problem_classes:
        raise ValueError(f"Unknown problem '{problem_name}'. "
                         f"Available: {list(problem_classes.keys())}")
    ProblemClass = problem_classes[problem_name]
    problem = ProblemClass(n=n)


    # STARTING POINT

    x0, start_id = generate_single_starting_point(problem, config)


    # gestisco h per finite differences, in base a derivatives.mode
    mode = config['derivatives']['mode']   # 'exact', 'fd_hessian', 'fd_all'
    if mode == 'exact':
        h = None
        relative = False
    else:
        # prendi il primo k e il primo flag relative dalla lista
        k_exp = config['derivatives']['h_exponents'][0]      # es. 4 → h = 1e-4
        h = 10.0 ** (-k_exp)
        relative = bool(config['derivatives']['relative'][0])

    all_results = {}

    for start_id, x0 in enumerate(starts):
        # Modified Newton
        if 'mn' in methods:
            res_mn = solve_modified_newton(problem, x0, config, h=h, relative=relative)
            key_mn = (problem_name, n, mode, 'mn', start_id)
            all_results[key_mn] = res_mn

        # Truncated Newton
        if 'tn' in methods:
            res_tn = solve_truncated_newton(problem, x0, config, h=h, relative=relative)
            key_tn = (problem_name, n, mode, 'tn', start_id)
            all_results[key_tn] = res_tn

    return all_results


if __name__ == "__main__":
    # 1) Leggi il config dell’esperimento corrente
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2) Esegui SOLO l’esperimento descritto in config.yaml
    all_results = main(config)

    # 3) Genera tabelle + figure per QUESTO esperimento
    #    (tutto va in output/tables e output/figures)
    postprocess(all_results, config, problem_classes)
