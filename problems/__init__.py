from problems.quad import make_quadratic
from problems.broyden_tridiagonal import BroydenTridiagonal

problem_classes = {
    "quad": make_quadratic,           # not a class yet
    "broyden_tridiagonal": BroydenTridiagonal
}

def get_problem_class(name: str):
    """Return the problem class corresponding to its string identifier."""
    try:
        return problem_classes[name]
    except KeyError:
        raise ValueError(
            f"Unknown problem '{name}'. Available problems: {list(problem_classes.keys())}"
        )