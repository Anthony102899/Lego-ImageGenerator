from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/LEGO_models/full_models/miniheads/41485 - Finn - Copy.mpd")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test.ldr"), debug=True)
    structure_graph = ConnectivityGraph(bricks)
    print(structure_graph.to_json())
    structure_graph.show()
