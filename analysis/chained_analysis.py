import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Chained Serpentine (n=2)
# -------------------------

def g_fun(x):
    """g(x) = 2x / (1 + x^2)"""
    return 2.0 * x / (1.0 + x * x)

def gprime_fun(x):
    """g'(x) = 2(1 - x^2) / (1 + x^2)^2"""
    xx = x * x
    return 2.0 * (1.0 - xx) / (1.0 + xx) ** 2

def objective_F(x0, x1):
    r0 = 10.0 * (g_fun(x0) - x1)
    s0 = x0 - 1.0
    return 0.5 * (r0 * r0 + s0 * s0)

def grad_exact_2d(x0, x1):
    r0 = 10.0 * (g_fun(x0) - x1)
    s0 = x0 - 1.0
    a0 = 10.0 * gprime_fun(x0)
    b  = -10.0
    g0 = r0 * a0 + s0
    g1 = r0 * b
    return np.array([g0, g1])

# -------------------------
# Plot settings
# -------------------------
x0_min, x0_max = -2.0, 2.0
x1_min, x1_max = -2.0, 2.0

# Grid for contours
N = 300
X0, X1 = np.meshgrid(np.linspace(x0_min, x0_max, N),
                     np.linspace(x1_min, x1_max, N))
Z = objective_F(X0, X1)

# Fewer levels -> contours more spaced (keep log-spacing for valley visibility)
Z_max = float(np.max(Z))
levels = np.geomspace(1e-3, Z_max, 14)  # << fewer lines (more distant)

fig, ax = plt.subplots(figsize=(9, 6))
cs = ax.contour(X0, X1, Z, levels=levels)
ax.clabel(cs, inline=True, fontsize=7)

ax.set_title("Chained Serpentine (n=2): contours + descent vectors")
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.set_xlim(x0_min, x0_max)
ax.set_ylim(x1_min, x1_max)

# Valley curve: x1 = g(x0)
x0_line = np.linspace(x0_min, x0_max, 500)
ax.plot(x0_line, g_fun(x0_line), linewidth=2, label="valley: $x_1=g(x_0)$")

# Minimum (for n=2)
ax.plot([1.0], [g_fun(1.0)], marker="o", markersize=7, label="minimum ~ (1,1)")

# -------------------------
# More points for gradients
# -------------------------
# Sample x0 positions
x0_pts = np.linspace(-1.8, 1.8, 11)

# Multiple offsets around the valley for each x0 (more arrows)
offsets = np.array([-0.9, -0.5, -0.25, 0.25, 0.5, 0.9])

P0 = np.repeat(x0_pts, offsets.size)
P1 = np.tile(offsets, x0_pts.size) + g_fun(P0)

# Compute gradients and plot ONLY -grad
G = np.array([grad_exact_2d(p0, p1) for p0, p1 in zip(P0, P1)])
G0, G1 = G[:, 0], G[:, 1]
Gn = np.sqrt(G0**2 + G1**2) + 1e-16

# Normalize arrow length so they’re readable
arrow_len = 0.18
U = (-G0 / Gn) * arrow_len   # -grad x-component
V = (-G1 / Gn) * arrow_len   # -grad y-component

ax.quiver(
    P0, P1, U, V,
    angles="xy", scale_units="xy", scale=1.0,
    width=0.0035,
    color="red",
    label=r"descent $-\nabla f$"
)

ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

