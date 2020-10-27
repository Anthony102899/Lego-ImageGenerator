import os
import trimesh
import numpy as np
import time
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance
from multiprocessing import Pool
from functools import partial
from bricks_modeling import config

def get_corner_pos(brick):
    bbox_ls = brick.get_bbox()
    cub_corner = []
    corner_transform = np.array([[1, 1, 1], [-1, -1, -1]])
    for bbox in bbox_ls:
        cuboid_center = np.array([bbox["Dimension"][0] / 2, bbox["Dimension"][1] / 2, bbox["Dimension"][2] / 2])
        cuboid_corner_relative = (np.tile(cuboid_center, (2, 1))) * corner_transform
        cub_corners_pos = np.array(bbox["Rotation"] @ cuboid_corner_relative.transpose()).transpose() + np.array(bbox["Origin"])
        cub_corner.append(cub_corners_pos[0])
        cub_corner.append(cub_corners_pos[1])
    return cub_corner

def brick_inside(brick:BrickInstance, mesh):
    corner_pos = get_corner_pos(brick)
    corner_inside = mesh.contains(corner_pos)
    nearby_faces = trimesh.proximity.nearby_faces(mesh, [brick.trans_matrix[:, 3][:3]])
    if corner_inside.all():
        return True, nearby_faces[0][0]
    return False, -1

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def check_brick(brick, mesh, colors_rgb):
    inside, nearby_face = brick_inside(brick, mesh)
    if inside:
        nearby_color = colors_rgb[nearby_face]
        nearby_hex = RGB_to_Hex(nearby_color)
        new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
        return new_brick
    return

def get_color(mesh):
    if type(mesh.visual) == trimesh.visual.ColorVisuals:
        colors_rgb = mesh.visual.face_colors
    else:
        colors_rgb = mesh.visual.to_color()
        if colors_rgb.defined:
            colors_rgb = colors_rgb.face_colors
        else:
            print("No valid color, using white!")
            colors_rgb = np.c_[np.ones((len(mesh.faces),3)) * 255, np.array([255 for face in mesh.faces])]
    return colors_rgb

def crop_brick(mesh, tile_set, scale):
    V = (mesh.vertices) * scale
    colors_rgb = get_color(mesh)
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    with Pool(20) as p:
        result_crop = p.map(partial(check_brick, mesh=mesh, colors_rgb=colors_rgb), tile_set)
    result_crop = [b for b in result_crop if b]
    return result_crop

if __name__ == "__main__":
    bricks = read_bricks_from_file("./debug/3005.ldr")
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/candle.ply")
    tile_path = os.path.join(os.path.dirname(__file__), 
                "super_graph/3005+4733+3024+54200+3070+59900/3005+4733+3023+3024+54200+3070+59900 3 n=4134 t=524.41.ldr")
    tile_set = read_bricks_from_file(tile_path)
    print("#bricks in tile: ", len(tile_set))
    mesh = trimesh.load_mesh(obj_path)
    print(mesh.is_watertight)

    if not type(mesh) == trimesh.Trimesh:
        mesh = mesh.dump(True)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    debugger = MyDebugger("crop")

    scale = float(input("Enter scale of obj: "))
    start_time = time.time()
    result = crop_brick(mesh, tile_set, scale)
    end_time = time.time()
    print(f"resulting LEGO model has {len(result)} bricks")

    _, filename = os.path.split(obj_path)
    filename = (filename.split("."))[0]
    _, tilename = os.path.split(tile_path)
    tilename = ((tilename.split("."))[0]).split(" ")
    tilename = tilename[0] + " " + tilename[1]
    write_bricks_to_file(result, file_path=debugger.file_path(f"n{filename} s={scale} n={len(result)} {tilename} t={round(end_time - start_time, 2)}.ldr"))