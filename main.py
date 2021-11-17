from bricks_modeling.bricks.brickinstance import get_corner_pos
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from util.debugger import MyDebugger
from visualization.model_visualizer import visualize_3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

if __name__ == "__main__":
    debugger = MyDebugger("test")
    # bricks = read_bricks_from_file("./debug/2021-11-15_19-49-14_LEGO0/LEGO0 b=24 ['3024', '3020', '3023', '3710', '43722', '43723'] .ldr")
    # bricks = read_bricks_from_file("./data/full_models/cube7.ldr")
    bricks = read_bricks_from_file("./solvers/generation_solver/['43723'] base=12 n=290 r=1.ldr")
    structure_graph = ConnectivityGraph(bricks)
    # graph = AdjacencyGraph(bricks)
    for brick in bricks:
        # bbox_ls = brick.get_col_bbox()
        corner_ls = np.array(get_corner_pos(brick, True))
        selected_ls = corner_ls[corner_ls[:,1] == 1][:,[0,2]]
        # plt.plot(selected_ls[:,0], selected_ls[:,1], marker="o", color="blue")
        fig, ax = plt.subplots()
        # plt.plot(Polygon(selected_ls, facecolor='k'))
        ax.add_patch(Polygon(selected_ls, facecolor='k'))
        ax.set_xlim([0, 300])
        ax.set_ylim([0, 300])
        plt.show()
    print(structure_graph.to_json())
    # structure_graph.show()

    points = [b.get_translation() for b in structure_graph.bricks]

    edges = [e["node_indices"] for e in structure_graph.connect_edges]
    visualize_3D(points, lego_bricks=bricks, edges=edges, show_axis=True)