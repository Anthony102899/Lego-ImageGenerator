#%%
import numpy as np
from sympy import symbols
from sympy.matrices import Matrix, zeros
from sympy import expand, collect, O

t_x, t_y, t_z = sympy.symbols("t_x t_y t_z")
x0_x, x0_y, x0_z = sympy.symbols("x0_x x0_y x0_z")
x1_x, x1_y, x1_z = sympy.symbols("x1_x x1_y x1_z")
x2_x, x2_y, x2_z = sympy.symbols("x2_x x2_y x2_z")

delta_t_x, delta_t_y, delta_t_z = sympy.symbols("delta_t_x delta_t_y delta_t_z")
delta_x0_x, delta_x0_y, delta_x0_z = sympy.symbols("delta_x0_x delta_x0_y delta_x0_z")
delta_x1_x, delta_x1_y, delta_x1_z = sympy.symbols("delta_x1_x delta_x1_y delta_x1_z")
delta_x2_x, delta_x2_y, delta_x2_z = sympy.symbols("delta_x2_x delta_x2_y delta_x2_z")

delta_t = Matrix([delta_t_x, delta_t_y, delta_t_z])
delta_x0 = Matrix([delta_x0_x, delta_x0_y, delta_x0_z])
delta_x1 = Matrix([delta_x1_x, delta_x1_y, delta_x1_z])
delta_x2 = Matrix([delta_x2_x, delta_x2_y, delta_x2_z])

t = Matrix([t_x, t_y, t_z])
x0 = Matrix([x0_x, x0_y, x0_z])
x1 = Matrix([x1_x, x1_y, x1_z])
x2 = Matrix([x2_x, x2_y, x2_z])

def compute_transform(t, x0, x1, x2):
    u1 = (x1 - x0).normalized()
    u2 = (x2 - x0).normalized()
    u3 = u1.cross(u2)

    T = zeros(3, 3)
    T[:, 0] = u1
    T[:, 1] = u2
    T[:, 2] = u3
    return T

x0_prime = x0 + delta_x0
x1_prime = x1 + delta_x1
x2_prime = x2 + delta_x2
t_prime = t + delta_t

T = compute_transform(t, x0, x1, x2)
T_prime = compute_transform(t_prime, x0_prime, x1_prime, x2_prime)

# %%
