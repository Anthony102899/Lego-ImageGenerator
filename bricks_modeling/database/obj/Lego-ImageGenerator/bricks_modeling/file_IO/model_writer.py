from typing import List
from bricks_modeling.bricks.brickinstance import BrickInstance
import random
from bricks_modeling.bricks.model import Model
from bricks_modeling.file_IO.util import to_ldr_format

def write_bricks_to_file(bricks: List[BrickInstance], file_path, debug=False):
    file = open(file_path, "a")
    ldr_content = "\n0 STEP\n".join([brick.to_ldraw() for brick in bricks])
    if debug: # output the connection points
        conn_point = "\n0 STEP\n".join(
            [c.to_ldraw() for brick in bricks for c in brick.get_current_conn_points()]
        )
        ldr_content = ldr_content + "\n" + conn_point
    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")


def write_bricks_to_file_with_steps(bricks: List[BrickInstance], file_path):
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

def write_model_to_file(model: Model, file_path):
    file = open(file_path, "a")
    ldr_content = ""

    for group_name in model.group_names:
        current_group = model.groups[group_name]
        ldr_content += f"0 FILE {group_name}\n"
        for brick_step in current_group.brick_steps:
            ldr_content += "0 STEP\n"
            # output all bricks
            for b in brick_step.bricks:
                ldr_content += (b.to_ldraw() + "\n")
            # output all subgroups
            for i in range(len(brick_step.subgroup_names)):
                name      = brick_step.subgroup_names[i]
                color     = brick_step.subgroups_colors[i]
                trans_mat = brick_step.subgroups_transformation[i]
                ldr_content += (to_ldr_format(color, trans_mat, name) + "\n")
        ldr_content += f"0 NOFILE\n"

    file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")
