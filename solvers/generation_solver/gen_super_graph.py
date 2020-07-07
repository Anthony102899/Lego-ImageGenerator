from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import form_complete_graph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np

brick_IDs = ["3004", "4070", # cuboid
             "4287", # slope
             "3070", # plate
             "3062", # round
             ]

def get_brick_templates(brick_IDs):
    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    for id in brick_IDs:
        assert id in template_ids
        brick_idx = template_ids.index(id)
        brickInstance = BrickInstance(
            brick_templates[brick_idx], np.identity(4, dtype=float)
        )
        bricks.append(brickInstance)

    return bricks



if __name__ == "__main__":
    debugger = MyDebugger("test")
    tile_set = get_brick_templates(brick_IDs)
    for num_rings in range(1, 2):
        tiles = form_complete_graph( # TODO: random orient the base_tile to test
            num_rings, base_tile=tile_set[0], tile_set=tile_set
        )  # including base tile
        print(f"number of tiles neighbours in ring{num_rings}:", len(tiles))
        write_bricks_to_file(
            tiles, file_path=debugger.file_path(f"test{num_rings}.ldr")
        )
