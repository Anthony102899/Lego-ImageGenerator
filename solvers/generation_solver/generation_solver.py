from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/miniheads/standard_core.mpd")
    write_bricks_to_file(
        bricks, file_path=debugger.file_path("test_single_brick.ldr"), debug=False
    )
    structure_graph = ConnectivityGraph(bricks)
    # print(structure_graph.to_json())
    # structure_graph.show()
