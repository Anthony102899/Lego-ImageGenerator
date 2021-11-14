from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
import numpy as np
from util.debugger import MyDebugger
from solvers.rigidity_solver.algo_core import solve_rigidity
from solvers.rigidity_solver.internal_structure import structure_sampling
import visualization.model_visualizer as vis
from solvers.rigidity_solver.eigen_analysis import get_motions, get_weakest_displacement
import solvers.rigidity_solver.test_cases.cases_3D as cases3d
from solvers.rigidity_solver.algo_core import rigidity_matrix,spring_energy_matrix_accelerate_3D
from util.geometry_util import trivial_basis,clear_redundance_vecs,clear_trivial_motion


if __name__ == "__main__":
    debugger = MyDebugger("test")

    bricks, points, edges, abstract_edges, points_on_brick = cases3d.lego_models("example1")

    # is_rigid, eigen_pairs = solve_rigidity(points, edges + abstract_edges, dim=3)

    K = spring_energy_matrix_accelerate_3D(points,edges,abstract_edges)

    w,V = np.linalg.eigh(K)
    V_normalized = map(lambda v: v / np.linalg.norm(v), V.T)
    eigen_pairs = sorted(list(zip(w, V_normalized)), key=lambda pair: pair[0])
    non_zero_eigenspace = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if abs(e_val) >= 1e-7]
    zero_motions = [(e_val, e_vec) for e_val, e_vec in eigen_pairs if
                    abs(e_val) < 1e-7]
    trivial_vec = trivial_basis(points, 3)
    basis = list(zip(*zero_motions))
    non_zero_eigenvecs = np.array(basis[1])
    ortho_basis = clear_redundance_vecs(non_zero_eigenvecs)
    after_clear = clear_redundance_vecs(clear_trivial_motion(ortho_basis, trivial_vec))
    F = K @ non_zero_eigenspace[0][1]
    print("the degree of freedom:", np.linalg.matrix_rank(after_clear))
    if np.linalg.matrix_rank(after_clear) == 6:
        is_rigid = True
    else:
        is_rigid = False


    if is_rigid:

        vis.visualize_3D(points, lego_bricks=bricks, arrows=(after_clear[0]).reshape(-1, 3))
    else:
        vis.visualize_3D(points, lego_bricks=bricks, arrows=(non_zero_eigenspace[0][1]).reshape(-1, 3))


