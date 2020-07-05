import trimesh
import numpy as np
import open3d as o3d
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file

obj_path = "./data/super_graph/bunny.obj"
tile_set = read_bricks_from_file("./data/super_graph/3004-ring7.ldr")

def brick_inside(brick, mesh):
    cpoints = brick.get_current_conn_points()
    cpoint_pos = list(map(lambda cp: list(cp.pos), cpoints))
    cpoint_inside = mesh.contains(cpoint_pos)
    if cpoint_inside.all():
        return True

def crop_brick(mesh, tile_set, scale):
    V = (mesh.vertices) * scale
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    result_crop = []
    for brick in tile_set:
        if brick_inside(brick, mesh):
            result_crop.append(brick)
    return result_crop

def get_color(mesh):
    colors = []
    face_num = len(mesh.faces)
    face_color = list(map(lambda f_color: list(trimesh.visual.color.to_rgba(f_color)), mesh.visual.face_colors))
    for face in range(face_num):
        colors.append(face_color[face])
    return colors

if __name__ == "__main__":
    scale = float(input("Enter scale of obj: "))
    mesh = trimesh.load(obj_path)
    result = crop_brick(mesh, tile_set, scale)
    print(f"resulting LEGO model has {len(result)} bricks")

    debugger = MyDebugger("test")
    write_bricks_to_file(result, file_path=debugger.file_path(f"lego-{scale}.ldr"))