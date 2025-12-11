import numpy as np

class ExtendedCraggLevy:

    def __init__(self, n):
        if n < 2:
            raise ValueError("Extended Cragg Levy requires n >= 2")

        self.n = n
        self.x0 = self.standard_starting_point()
        pass


    def standard_starting_point(self):
        x = np.empty(self.n, dtype=float)
        for i in range(self.n):
            if (i + 1) % 4 == 1:
                x[i] = 1.0
            else:
                x[i] = 2.0
        return x

    # RESIDUAL f
    def cragg_levy_residual(self, x):
        x = np.asarray(x, dtype=float)
        n = self.n
        f = np.empty(n, dtype=float)

        for k in range(n):
            kk = k + 1        # using kk as mathematial index since k starts from 0
            r = kk % 4

            if r == 1:
                s = np.exp(x[k] - x[k + 1])
                f[k] = s * s
            elif r == 2:
                d = x[k] - x[k + 1]
                f[k] = 10.0 * d**3
            elif r == 3:
                t = np.tan(x[k] - x[k + 1])
                f[k] = t * t
            else:   # r == 0
                f[k] = x[k] - 1.0

        return f

    # OBJECTIVE VALUE
    def f(self,x):
        """
        F(x) = 0.5 * ||f(x)||^2
        """
        f = self.cragg_levy_residual(x)
        return 0.5 * float(np.dot(f, f))


    # GRADIENT
    def grad_exact(self, x):
        x = np.asarray(x, dtype=float)
        n = self.n

        g = np.zeros(n, dtype=float)

        for k in range(n):
            kk = k + 1
            r = kk % 4

            if r == 1:
                s = np.exp(x[k]) - x[k + 1]
                f_k = s * s

                df_dx_k = 2.0 * s * np.exp(x[k])
                df_dx_kp1 = -2.0 * s

                g[k] += f_k * df_dx_k
                g[k + 1] += f_k * df_dx_kp1

            elif r == 2:
                d = x[k] - x[k + 1]
                f_k = 10.0 * d ** 3

                df_dx_k = 30.0 * d**2
                df_dx_kp1 = -30.0 * d**2

                g[k] += f_k * df_dx_k
                g[k + 1] += f_k * df_dx_kp1

            elif r == 3:
                theta = x[k] - x[k + 1]
                t = np.tan(theta)
                f_k = t * t

                df_dt_theta = 2.0 * t * (1 + t * t)
                df_dx_k = df_dt_theta
                df_dx_kp1 = -df_dt_theta

                g[k] += f_k * df_dx_k
                g[k + 1] += f_k * df_dx_kp1

            else:   # r == 0
                f_k = x[k] - 1.0
                df_dx_k = 1.0
                g[k] += f_k * df_dx_k

        return g






