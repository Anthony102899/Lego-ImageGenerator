from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from solvers.rigidity_solver.algo_core import solve_rigidity
from solvers.rigidity_solver.internal_structure import structure_sampling
import visualization.model_visualizer as vis
from solvers.rigidity_solver.eigen_analysis import get_motions, get_weakest_displacement
import solvers.rigidity_solver.test_cases.cases_3D as cases3d


if __name__ == "__main__":
    debugger = MyDebugger("test")

    bricks, points, edges, abstract_edges, points_on_brick = cases3d.case_normal("hinged_L")

    is_rigid, eigen_pairs = solve_rigidity(points, edges + abstract_edges, dim=3)

    if is_rigid:
        print("Rigid structure!")
        vec, value = get_weakest_displacement(eigen_pairs, dim=3)
        vis.visualize_3D(points, lego_bricks=bricks, edges=edges, arrows=vec)
    else:
        print("non-Rigid structure!")
        motion_vecs = get_motions(eigen_pairs, points, dim=3)
        vis.visualize_3D(points, lego_bricks=bricks, edges=edges, arrows=motion_vecs[0])


