from problems.broyden_tridiagonal import BroydenTridiagonal
from problems.chained_serpentine import ChainedSerpentine

problem_classes = {
    "broyden_tridiagonal": BroydenTridiagonal,
    "chained_serpentine": ChainedSerpentine
}

def get_problem_class(name: str):
    """Return the problem class corresponding to its string identifier."""
    try:
        return problem_classes[name]
    except KeyError:
        raise ValueError(
            f"Unknown problem '{name}'. Available problems: {list(problem_classes.keys())}"
        )