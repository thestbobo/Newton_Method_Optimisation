import numpy as np

class ExtendedPowelSingular:
    """
    Extended Powell singular function.

    Dimension n must be a multiple of 4.
    F(x) = 0.5 * sum_{k=1}^n f_k(x)^2
    with 4 residuals per block:
        f1 = x1 + 10 x2
        f2 = sqrt(5) (x3 - x4)
        f3 = (x2 - 2 x3)^2
        f4 = sqrt(10) (x1 - x4)^2
    where (x1, x2, x3, x4) is a block of 4 consecutive components.
    """

    def __init__(self, n):
        assert n % 4 == 0, "Extended Powel: n must be a multiple of 4"
        self.n = n
        self.m = n / 4

        # coefficients
        self.c1 = 10.0
        self.c2 = np.sqrt(5.0)
        self.c3 = np.sqrt(10.0)
        pass


    def standard_starting_point(self):
        x0 = np.zeros(self.n, dtype=float)
        pattern = np.array([3.0, -1.0, 0.0, 1.0])

        return x0
