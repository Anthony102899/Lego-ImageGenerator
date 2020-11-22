import os
import numpy as np
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from util.geometry_util import get_random_transformation
from bricks_modeling.bricks.brickinstance import BrickInstance
from typing import List
from bricks_modeling.file_IO.model_reader import read_bricks_from_file

def draw_bbox(bbox,i):
    origin = bbox["Origin"]
    rot_mat = bbox["Rotation"]
    scaling = np.identity(3)
    row, col = np.diag_indices(scaling.shape[0])
    scaling[row, col] = np.array([bbox["Dimension"][0]/2, bbox["Dimension"][1], bbox["Dimension"][2]/2])
    matrix = rot_mat @ scaling
    offset = matrix @ np.array([0, -0.5, 0])
    text = (
            f"1 {1+i} {origin[0] + offset[0]} {origin[1] + offset[1]} {origin[2] + offset[2]} "
            + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
            + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
            + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
            + "box5.dat")
    return text

def write_bricks_w_bbox(bricks: List[BrickInstance], file_path):
    file = open(file_path, "a")
    for brick in bricks:
        bbox = brick.get_col_bbox()
        ldr_content = "\n0 STEP\n".join([brick.to_ldraw()])

        bbox_text = "\n".join([draw_bbox(bbox[i],i) for i in range(len(bbox))])
        ldr_content = ldr_content + "\n" + bbox_text + "\n"
        file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")

if __name__ == "__main__":
    #mode = int(input("Enter mode: "))
    mode = 1
    if mode == 1:
        debugger = MyDebugger("drawbbox")
        file_path = "./debug/1 3023+3024.ldr"
        bricks = read_bricks_from_file(file_path)
        _, filename = os.path.split(file_path)
        filename = (filename.split("."))[0]
        write_bricks_w_bbox(bricks, file_path=debugger.file_path(f"{filename}_test.ldr"))

    elif mode == 2:
        debugger = MyDebugger("drawallbbox")
        brick_templates, template_ids = get_all_brick_templates()
        for template in brick_templates:
            brickInstance = BrickInstance(template, np.identity(4, dtype=float), 15)
            write_bricks_w_bbox([brickInstance], file_path=debugger.file_path(f"{template.id}_test.ldr"))