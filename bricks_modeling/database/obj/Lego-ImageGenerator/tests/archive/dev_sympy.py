#%%
import sympy
import numpy as np
from sympy import symbols, collect, expand
from sympy.matrices import Matrix, zeros
from sympy import expand, collect, O

from tqdm import tqdm

from functools import reduce
from itertools import product

# %%
def listsubs(expr, vars, values):
    assert len(vars) == len(values)
    for var, value in zip(vars, values):
        expr = expr.subs(var, value)
    return expr

def dictsubs(expr, mapping):
    for var, value in mapping.items():
        expr = expr.subs(var, value)
    return expr

t_x, t_y, t_z = symbols("t_x t_y t_z")
x0_x, x0_y, x0_z = symbols("x0_x x0_y x0_z")
x1_x, x1_y, x1_z = symbols("x1_x x1_y x1_z")
x2_x, x2_y, x2_z = symbols("x2_x x2_y x2_z")

delta_t_x, delta_t_y, delta_t_z = symbols("delta_t_x delta_t_y delta_t_z")
delta_x0_x, delta_x0_y, delta_x0_z = symbols("delta_x0_x delta_x0_y delta_x0_z")
delta_x1_x, delta_x1_y, delta_x1_z = symbols("delta_x1_x delta_x1_y delta_x1_z")
delta_x2_x, delta_x2_y, delta_x2_z = symbols("delta_x2_x delta_x2_y delta_x2_z")

delta_t = Matrix([delta_t_x, delta_t_y, delta_t_z])
delta_x0 = Matrix([delta_x0_x, delta_x0_y, delta_x0_z])
delta_x1 = Matrix([delta_x1_x, delta_x1_y, delta_x1_z])
delta_x2 = Matrix([delta_x2_x, delta_x2_y, delta_x2_z])

t = Matrix([t_x, t_y, t_z])
x0 = Matrix([x0_x, x0_y, x0_z])
x1 = Matrix([x1_x, x1_y, x1_z])
x2 = Matrix([x2_x, x2_y, x2_z])

norm_1 = symbols("norm_1")
norm_2 = symbols("norm_2")

# norm_1 = (x1 - x0).norm()
# norm_2 = (x2 - x0).cross(u1).norm()

# x0 = x0.subs(x0_x, 0).subs(x0_y, 0).subs(x0_z, 0)
# x1 = x1.subs(x1_x, 0).subs(x1_y, 0).subs(x1_z, 1)
# x2 = x2.subs(x2_x, 0).subs(x2_y, 1).subs(x2_z, 0)
# t = t.subs(t_x, 1).subs(t_y, 0).subs(t_z, 0)

# u1 = (x1 - x0) / norm_1
# u2 = (x2 - x0).cross(u1) / norm_2

x0_prime = x0 + delta_x0
x1_prime = x1 + delta_x1
x2_prime = x2 + delta_x2
t_prime = t + delta_t

u1 = (x1 - x0) / norm_1
u2 = (x2 - x0).cross(u1) / norm_2
u3 = u1.cross(u2)

T = zeros(3, 3)
T[:, 0] = u1
T[:, 1] = u2
T[:, 2] = u3

u1_prime = (x1_prime - x0_prime) / norm_1
u2_prime = (x2_prime - x0_prime).cross(u1_prime) / norm_2
u3_prime = u1_prime.cross(u2_prime)

T_prime = zeros(3, 3)
T_prime[:, 0] = u1_prime
T_prime[:, 1] = u2_prime
T_prime[:, 2] = u3_prime

# b = T @ (t - x0)
b = zeros(3, 1)
b[0] = (t - x0).dot(u1)
b[1] = (t - x0).dot(u2)
b[2] = (t - x0).dot(u3)
# b = T @ (t - x0)
# b_x, b_y, b_z = symbols("b_x b_y b_z")
# b = Matrix([b_x, b_y, b_z])
# b_prime = T_prime @ (t_prime - x0_prime)
b_prime = T_prime.T @ (t_prime - x0_prime)
# b_prime = zeros(3, 1)
# b_prime[0] = (t_prime - x0_prime).dot(u1_prime)
# b_prime[1] = (t_prime - x0_prime).dot(u2_prime)
# b_prime[2] = (t_prime - x0_prime).dot(u3_prime)
delta_b = b_prime - b

deltas = [
    delta_x0_x, delta_x0_y, delta_x0_z,
    delta_x1_x, delta_x1_y, delta_x1_z,
    delta_x2_x, delta_x2_y, delta_x2_z,
    delta_t_x, delta_t_y, delta_t_z,
]

varlist = [
    x0_x, x0_y, x0_z, x1_x, x1_y, x1_z, x2_x, x2_y, x2_z, t_x, t_y, t_z,
    norm_1, norm_2,
] + deltas

#%%
constant_values = np.array([
    0, 1, 0,
    0, 1, 1,
    0, 2, 0,
    1, 0, 1,
])
x1_minus_x0 = np.array(constant_values[3: 6]) - np.array(constant_values[0: 3])
v1 = x1_minus_x0
u1 = v1 / np.linalg.norm(v1)
x2_minus_x0 = np.array(constant_values[6: 9]) - np.array(constant_values[0: 3])
v2 = np.cross(x2_minus_x0, u1)
norm_values = [
    np.linalg.norm(v1),
    np.linalg.norm(v2),
]
delta_values = np.array([
    0, 0, 1,
    -1, 1, 1,
    0, 0, 2,
    1, -1, 1,
]) 

context = {var: value for var, value in zip(varlist, np.hstack((constant_values, norm_values, delta_values)))}
dictsubs(delta_b, context)

#%%

O_deltas = reduce(lambda x, y: x + y, [O(d) for d in deltas])

expanded_delta_b = expand(delta_b)
print("expanded delta_b")
projection_matrix = zeros(3, 12)
projection_offset = zeros(3)

for i, j in tqdm(product(range(3), range(12))):
    coeff = collect(expanded_delta_b[i], deltas[j]).coeff(deltas[j], 1)
    order_one_coeff = (coeff + O_deltas).removeO()

    projection_matrix[i, j] = sympy.simplify(order_one_coeff)

dictsubs(projection_matrix, context)
#%%
dictsubs(projection_matrix @ Matrix(deltas), context)

#%%

# for var_index in [3, 4, 5]:
#     valuelist = constant_values[:9] + constant_values[var_index * 3: var_index * 3 + 3] + \
#                 norm_values + \
#                 delta_values[:9] + delta_values[var_index * 3: var_index * 3 + 3]
#     results = listsubs(projection_matrix, varlist, valuelist)
#     results = results @ Matrix(deltas)
#     results = listsubs(results, varlist, valuelist)
#     print(results)
#     print("\n\n")

# %%
for i, j in product(range(3), range(12)):
    print(f"projection_matrix[{i}, {j}] = {projection_matrix[i, j]}")
#%%

with open("dev-projection-matrix.txt", "w") as fp:
    for i, j in product(range(3), range(12)):
        print(f"projection_matrix[{i}, {j}] = {projection_matrix[i, j]}", file=fp)
