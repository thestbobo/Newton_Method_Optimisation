from problems.quad import make_quadratic
from problems.broyden_tridiagonal import BroydenTridiagonal

PROBLEM_CLASSES = {
    "quad": make_quadratic,           # not a class yet
    "broyden_tridiagonal": BroydenTridiagonal
}

def get_problem_class(name: str):
    """Return the problem class corresponding to its string identifier."""
    try:
        return PROBLEM_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown problem '{name}'. Available problems: {list(PROBLEM_CLASSES.keys())}"
        )