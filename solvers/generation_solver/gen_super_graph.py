import os
import sys

ROOT_DIR = os.path.abspath('/Users/wuyifan/lego-solver')
sys.path.append(ROOT_DIR)

from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import form_complete_graph

tile_set = read_bricks_from_file("./data/single_part/3005.ldr")  # a list of bricks

if __name__ == '__main__':
    data = []
    for num_rings in range(1,2):
        tiles = form_complete_graph(num_rings, base_tile=tile_set[0], tile_set=tile_set)
