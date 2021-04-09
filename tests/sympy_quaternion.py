#%%
import sympy
import sys

from sympy import (
    Matrix,
    Quaternion,
    symbols,
    zeros, eye,
    expand, collect, O,
    solve_linear_system
)

from sympy.abc import x, y, z
#%%
theta = symbols("theta")
vx, vy, vz = symbols("v_x, v_y, v_z")

rot_mat = Quaternion.from_axis_angle((vx, vy, vz), theta).to_rotation_matrix()
rot_mat = Matrix(3, 3, symbols("a_:3:3"))

aug_mat = Matrix.hstack(rot_mat, zeros(3, 1))

sol = solve_linear_system(aug_mat, vx, vy, vz)


# Quaternion.from_rotation_matrix(A)
#%%
x_list = symbols(" ".join(f"x{i}(:3)" for i in range(6)))
x = Matrix(x_list).reshape(6, 3)

delta_x_list = symbols(" ".join(f"delta_x{i}(:3)" for i in range(6)))
delta_x = Matrix(delta_x_list).reshape(6, 3)

x_prime = x + delta_x
#%%
a = x.row(1) - x.row(0)
# norm_a = a.norm()
norm_a = symbols("norm_a")

u = a / norm_a

b = a.cross(x.row(2) - x.row(0)) 
# norm_b = b.norm()
norm_b = symbols("norm_b")

v = b / norm_b

w = u.cross(v)

# u = T v -> T transform world vector v into local vector u
T = Matrix([u, v, w]) 

#%%
a_prime = x_prime.row(1) - x_prime.row(0)
# norm_a = a.norm()
# norm_a = symbols("norm_a")

u_prime = a_prime / norm_a

b_prime = a_prime.cross(x_prime.row(2) - x_prime.row(0)) 
# norm_b = b.norm()
# norm_b = symbols("norm_b")

v_prime = b_prime / norm_b

w_prime = u_prime.cross(v_prime)

# u = T_prime v -> T_prime transform world vector v into local vector u (after perturbation)
T_prime = Matrix([u_prime, v_prime, w_prime]) 

#%%
T_diff = T_prime @ T.transpose()
rot = T_diff[:3, :3]
#%%
# Property of rotation matrix R: R * axis = axis
#                           => (R - I) * axis = 0
axis = symbols("u_x, u_y, u_z")
aug_mat = Matrix.hstack(rot - eye(3), zeros(3, 1))
solve_linear_system(aug_mat, *axis)


#%%
from itertools import product
from functools import reduce

order_one_T_diff = zeros(3, 3)
expanded_T_diff = expand(T_diff)
O_two = reduce(lambda x, y: x + y, [O(m * n) for m, n in product(delta_x_list, repeat=2)])
O_three = reduce(lambda x, y: x + y, [O(m * n * o) for m, n, o in product(delta_x_list, repeat=3)])
O_four = reduce(lambda x, y: x + y, [O(m * n * o * p) for m, n, o, p in product(delta_x_list, repeat=4)])
O_all = O_two + O_three + O_four
for i, j in product(range(3), range(3)):
    order_one_T_diff[i, j] = (expanded_T_diff[i, j] + O_all).removeO()

#%%
quat = Quaternion.from_rotation_matrix(order_one_T_diff)
