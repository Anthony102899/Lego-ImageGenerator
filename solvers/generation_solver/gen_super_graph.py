from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import form_complete_graph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file

tile_set = read_bricks_from_file("./data/single_part/3004.ldr")  # a list of bricks

if __name__ == '__main__':
    debugger = MyDebugger("test")
    for num_rings in range(1,2):
        tiles = form_complete_graph(num_rings, base_tile=tile_set[0], tile_set=tile_set)  # including base tile
        print(f"number of tiles neighbours in ring{num_rings}", len(tiles))
        write_bricks_to_file(tiles, file_path=debugger.file_path(f"test{num_rings}.ldr"))