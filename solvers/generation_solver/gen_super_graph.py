from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import form_complete_graph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

brick_IDs = ["3004",
             # "4070", # cuboid
             # "4287", # slope
             # "3070", # plate
             # "3062", # round
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

def gen_random_rotation():
    rand_rot_axis = np.random.rand(3)
    rot = R.from_rotvec(np.random.randint(0, 100) * rand_rot_axis)
    return rot.as_matrix()

if __name__ == "__main__":
    debugger = MyDebugger("test")
    brick_set = get_brick_templates(brick_IDs)

    seed_brick = copy.deepcopy(brick_set[0])
    trans_matrix = np.identity(4, dtype=float)
    trans_matrix[:3, :3] = gen_random_rotation()
    trans_matrix[:3, 3] = 5 * np.random.rand(3)
    seed_brick.trans_matrix = trans_matrix

    for num_rings in range(1, 3):
        tiles = form_complete_graph( # TODO: random orient the base_tile to test
            num_rings, base_tile=seed_brick, tile_set=brick_set
        )  # including base tile

        print(f"number of tiles neighbours in ring{num_rings}:", len(tiles))
        write_bricks_to_file(
            tiles, file_path=debugger.file_path(f"test{num_rings}.ldr")
        )
