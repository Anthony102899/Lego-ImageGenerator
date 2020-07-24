from typing import List
from bricks_modeling.bricks.brickinstance import BrickInstance
import random

def write_bricks_to_file(bricks: List[BrickInstance], file_path, debug=False):
    file = open(file_path, "a")
    ldr_content = "\n0 STEP\n".join([brick.to_ldraw() for brick in bricks])
    if debug:
        conn_point = "\n0 STEP\n".join(
            [c.to_ldraw() for brick in bricks for c in brick.get_current_conn_points()]
        )
        ldr_content = ldr_content + "\n" + conn_point
    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")


def write_bricks_to_file_for_instruction(bricks: List[BrickInstance], file_path):
    file = open(file_path, "a")
    ldr_content = ""
    step_count = 0
    for b in bricks:
        ldr_content += b.to_ldraw()
        step_count = step_count - 1
        if step_count <= 0:
            ldr_content += "\n0 STEP"
            step_count = random.randint(2,5)
        ldr_content += "\n"

    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")
