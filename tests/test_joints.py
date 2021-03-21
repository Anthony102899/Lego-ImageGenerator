from solvers.rigidity_solver.joints import *

from solvers.rigidity_solver.algo_core import solve_rigidity, spring_energy_matrix
from numpy import linalg as LA
from scipy.linalg import null_space
from numpy.linalg import cholesky, inv, matrix_rank

import util.geometry_util as geo_util
from visualization.model_visualizer import visualize_3D, visualize_hinges

from testcases import simple, tetra

model = tetra.square_pyramid_axes()
# axes = np.array([
#     [-0.6002, 0.2442, 0.7616],
#     [0.6295, 0.3026, 0.7157],
#     [0.2777, -0.2236, 0.9343],
#     [0.5090, 0.2976, 0.8077],
# )
# axes = np.array([[-0.67047648, 0.73671675, 0.08780502],
#  [-0.99870806, 0.04578672, 0.02204064],
#  [0.35756634, -0.71508136, 0.60067042],
#  [0.38999463, -0.34514895, 0.85368401]])

points = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])

axes = np.array([[0.75262934, -0.07756639, 0.65385972],
                 [0.37093456, -0.22086993, 0.9020111],
                 [0.6943383, 0.27738536, 0.66404193],
                 [0.31134789, -0.85239178, 0.42010802]])

model = np.array([[-0.98563743, -0.12848687, 0.10959004],
                  [-0.19268917, -0.07277234, 0.97855765],
                  [0.08891251, -0.0505588, 0.99475543],
                  [0.43746994, -0.55973823, 0.70378488]])
model = np.array([[-0.13048289, 0.28694951, 0.94901749],
 [-0.00646574,-0.05100298, 0.99867757],
 [ 0.02604103,-0.09707922, 0.99493592],
 [ 0.37766785,-0.43840374, 0.81557903]])
model = np.array([[ 0.85433678, 0.17596682,-0.48902387],
 [-0.60376715,-0.24374286, 0.75898264],
 [ 0.40050693,-0.36615589, 0.8399548 ],
 [ 0.69355212,-0.03185574, 0.71970179]
])
model = tetra.square(axes)
model.visualize()

model = tetra.square(axes)

dim = 3
points = model.point_matrix()
edges = model.edge_matrix()
A = model.constraint_matrix()

A = A if A.size != 0 else np.zeros((1, len(points) * dim))

trivial_motions = geo_util.trivial_basis(points, dim=3)
fixed_coordinates = np.zeros((len(model.beams[0].points) * 3, points.shape[0] * 3))
for r, c in enumerate(range(len(model.beams[0].points) * 3)):
    fixed_coordinates[r, c] = 1
# A = np.vstack((A, fixed_coordinates))
A = np.vstack((A, np.take(trivial_motions, [0, 1, 2, 3, 4, 5], axis=0)))

pivots = np.array([j.pivot_point for j in model.joints])
axes = np.array([j.axis for j in model.joints])
visualize_hinges(points, edges=edges, pivots=pivots, axes=axes)

M = spring_energy_matrix(points, edges, dim=dim)
print("M rank:", matrix_rank(M))

# mathmatical computation
B = null_space(A)
T = np.transpose(B) @ B
S = B.T @ M @ B

print("A shape:", A.shape)
print("T rank:", matrix_rank(T))
print("B rank:", matrix_rank(B))
print("S rank:", matrix_rank(S))
print("S shape", S.shape)

L = cholesky(T)
L_inv = inv(L)

Q = LA.multi_dot([L_inv.T, S, L_inv])
print("Q shape", Q.shape)
# compute eigenvalues / vectors
eigen_pairs = geo_util.eigen(Q, symmetric=True)
eigen_pairs = [(e_val, B @ e_vec) for e_val, e_vec in eigen_pairs]

# determine rigidity by the number of zero eigenvalues
zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) < 1e-6]
print("DoF:", len(zero_eigenspace))

trivial_motions = geo_util.trivial_basis(points, dim=3)

non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-8]

if len(zero_eigenspace) > 0:
    print("Non-rigid")
    for e, v in zero_eigenspace:
        arrows = v.reshape(-1, 3)
        print(e)
        visualize_3D(points, edges=edges, arrows=arrows)
        # visualize_3D(points, edges=edges)
else:
    print("rigid")
    # for e, v in non_zero_eigenspace:
    #     arrows = v.reshape(-1, 3)
    #     visualize_3D(points, edges=edges, arrows=arrows)
    #
    e, v = non_zero_eigenspace[0]
    print("smallest eigenvalue:", e)
    arrows = v.reshape(-1, 3)
    # visualize_3D(points, edges=edges)
    visualize_3D(points, edges=edges, arrows=np.where(np.isclose(arrows, 0), 0, arrows))