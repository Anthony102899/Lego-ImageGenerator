import sys
sys.path.append('.')

from util.debugger import MyDebugger
import visualization.model_visualizer as vis
from solvers.rigidity_solver.eigen_analysis import get_motions, get_weakest_displacement
import solvers.rigidity_solver.test_cases.cases_2D as cases2d
from solvers.rigidity_solver.algo_core import solve_rigidity

if __name__ == "__main__":
    debugger = MyDebugger("test")

    points, fixed_points_index, edges, joints = cases2d.case_11_1()

    trivial_motions, non_trivial_motions, non_zero_eigenspace = \
        solve_rigidity(points, edges, joints, fixed_points_idx=fixed_points_index, dim=2)

    print("# trivial motions: ", len(trivial_motions))
    print("# non-trivial motions: ", len(non_trivial_motions))
    print("# non_zero motions: ", len(non_zero_eigenspace))

    if len(non_trivial_motions) == 0: # rigid structure
        vec, value = get_weakest_displacement(non_zero_eigenspace, dim=2)
        print(f"worse case value: {value}")
        vis.visualize_2D(points, edges, vec*1)
        print("The structure is RIGID.")
    else:
        motion_vecs = get_motions(non_trivial_motions, points, dim=2)
        vis.visualize_2D(points, edges, motion_vecs[0])
        print("The structure is NOT RIGID.")
