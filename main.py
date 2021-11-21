from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from visualization.model_visualizer import visualize_3D

if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./debug/2021-11-21_21-30-38_heart/heart b=24 ['3024', '43722', '43723'] .ldr")
    # bricks = read_bricks_from_file("./data/full_models/cube7.ldr")
    structure_graph = ConnectivityGraph(bricks)
    print(structure_graph.to_json())
    # structure_graph.show()

    points = [b.get_translation() for b in structure_graph.bricks]

    edges = [e["node_indices"] for e in structure_graph.connect_edges]
    visualize_3D(points, lego_bricks=bricks, edges=edges, show_axis=True)