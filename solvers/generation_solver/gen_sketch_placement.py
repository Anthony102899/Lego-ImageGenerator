import os
import time

import numpy as np

from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.get_sketch import inspect
from solvers.generation_solver.tile_graph import find_brick_placements
from util.debugger import MyDebugger

brick_IDs = [# tile
             # plate 
             "3024",
             "3020",
             "3023",
             "3710",
             "43722",
             "43723"
             # other
            ]

def get_brick_templates(brick_IDs):
    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    for id in brick_IDs:
        assert id in template_ids
        brick_idx = template_ids.index(id)
        brickInstance = BrickInstance(
            brick_templates[brick_idx], np.identity(4, dtype=float))
        bricks.append(brickInstance)

    return bricks

def generate_new_plate(brick_set, base, num_rings, base_num):
    start_time = time.time()
    bricks = find_brick_placements(num_rings, base, tile_set=brick_set, sketch=True, base_num=base_num)
    print(f"generate finished in {round(time.time() - start_time, 2)}")
    print(f"number of tiles neighbours in ring{num_rings}:", len(bricks))
    return bricks

if __name__ == "__main__":
    debugger = MyDebugger("gen")
    brick_set = get_brick_templates(brick_IDs)
    base_path = os.path.join(os.path.dirname(__file__), "base 24.ldr")
    base = read_bricks_from_file(base_path)
    _, base_name = os.path.split(base_path)
    base_name = ((base_name.split(" "))[1]).split(".")[0]
    base_num = int(base_name)

    num_rings = int(input("Enter ring: "))
    bricks = generate_new_plate(brick_set, base=base, num_rings=num_rings, base_num=base_num)
    # inspect(bricks=bricks, bricks_only=True, basenum=2, depictbase=True, base=bricks[:2])
    write_bricks_to_file(
        bricks, file_path=debugger.file_path(f"{brick_IDs} base={base_name} n={len(bricks)} r={num_rings}.ldr"))
    print("done!")
