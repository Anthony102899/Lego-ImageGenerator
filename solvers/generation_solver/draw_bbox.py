import numpy as np
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from util.geometry_util import get_random_transformation
from bricks_modeling.bricks.brickinstance import BrickInstance
from typing import List
from bricks_modeling.file_IO.model_reader import read_bricks_from_file

def get_corner(corner_transform, bbox):
    rotation = [cuboid["Rotation"] for cuboid in bbox]
    cuboid_center = [np.array([cuboid["Dimension"][0] / 2, 
                               cuboid["Dimension"][1] / 2, 
                               cuboid["Dimension"][2] / 2]) for cuboid in bbox]  #(8,3)
    cuboid_corner_relative = [(np.tile(center, (4, 1))) * corner_transform for center in cuboid_center] #(8,4,3)
    centers = [np.tile(cuboid["Origin"], (4, 1)) for cuboid in bbox] #(8,4,3)
    cub_corners_pos = [(rotation[i] @ cuboid_corner_relative[i].transpose()).transpose() + centers[i] for i in range(len(centers))]
    return cub_corners_pos

def draw_box(box,i):
    text = (f"4 {1+i} "
            + f"{box[0][0]} {box[0][1]} {box[0][2]} "
            + f"{box[1][0]} {box[1][1]} {box[1][2]} "
            + f"{box[2][0]} {box[2][1]} {box[2][2]} "
            + f"{box[3][0]} {box[3][1]} {box[3][2]}")
    return text

def write_bricks_w_bbox(bricks: List[BrickInstance], file_path):
    file = open(file_path, "a")
    for brick in bricks:
        #print(brick_rot)
        bbox = brick.get_bbox()
        #print((bbox[0])["Origin"],"\n")
        #print((bbox[0])["Dimension"],"\n")
        ldr_content = "\n0 STEP\n".join([brick.to_ldraw()])

        corner_transform1 = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
        corner_transform2 = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
        corner_transform3 = np.array([[1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1]])
        corner_transform4 = np.array([[1, -1, 1], [1, -1, -1], [-1, -1, -1], [-1, -1, 1]])

        corners1 = get_corner(corner_transform1, bbox)
        box1 = "\n".join([draw_box(corners1[i], i) for i in range(len(corners1))])
        corners2 = get_corner(corner_transform2, bbox)
        box2 = "\n".join([draw_box(corners2[i], i) for i in range(len(corners2))])
        corners3 = get_corner(corner_transform3, bbox)
        box3 = "\n".join([draw_box(corners3[i], i) for i in range(len(corners3))])
        corners4 = get_corner(corner_transform4, bbox)
        box4 = "\n".join([draw_box(corners4[i], i) for i in range(len(corners4))])

        ldr_content = ldr_content + "\n" + box1+ "\n" + box2+ "\n" + box3+ "\n" + box4 + "\n"
        file.write(ldr_content)
    file.close()
    print(f"file {file_path} saved!")

if __name__ == "__main__":
    debugger = MyDebugger("drawbbox")
    #mode = int(input("Enter mode: "))
    mode = 1
    if mode == 1:
        file_path = "./debug/0 3024+54200.ldr"
        bricks = read_bricks_from_file(file_path)
        _, filename = os.path.split(file_path)
        filename = (filename.split("."))[0]
        write_bricks_w_bbox(bricks, file_path=debugger.file_path(f"{filename}_test.ldr"))

    elif mode == 2:
        brick_templates, template_ids = get_all_brick_templates()
        for template in brick_templates:
            brickInstance = BrickInstance(template, np.identity(4, dtype=float), 15)
            write_bricks_w_bbox([brickInstance], file_path=debugger.file_path(f"{template.id}_test.ldr"))