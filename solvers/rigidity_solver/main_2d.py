from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from solvers.rigidity_solver.algo_core import solve_rigidity
from solvers.rigidity_solver.internal_structure import structure_sampling
import visualization.model_visualizer as vis
from solvers.rigidity_solver.eigen_analysis import get_motions, get_weakest_displacement
import solvers.rigidity_solver.test_cases.cases_2D as cases2d


if __name__ == "__main__":
    debugger = MyDebugger("test")

    points, fixed_points_index, edges, abstract_edges = cases2d.case_8()

    is_rigid, eigen_pairs = solve_rigidity(points, edges + abstract_edges, fixed_points=fixed_points_index, dim=2)

    if is_rigid:
        vec, value = get_weakest_displacement(eigen_pairs, dim=2)
        print(f"worse case value: {value}")
        # print(vec)
        vis.visualize_2D(points, edges, vec)
    else:
        motion_vecs = get_motions(eigen_pairs, points, dim=2)
        vis.visualize_2D(points, edges, motion_vecs[0])

    print("The structure is", "rigid" if is_rigid else "not rigid.")