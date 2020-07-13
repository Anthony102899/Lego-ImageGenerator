import os
import trimesh
import numpy as np
import time
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance

def brick_inside(brick, mesh):
    cpoint_pos = list(map(lambda cp: list(cp.pos), brick.get_current_conn_points()))  # position of cpoints of *brick*
    cpoint_inside = mesh.contains(cpoint_pos)
    nearby_faces = trimesh.proximity.nearby_faces(mesh, [brick.trans_matrix[:, 3][:3]])
    if cpoint_inside.all():
        return True, nearby_faces[0][0]
    return False, -1

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def crop_brick(mesh, tile_set, scale):
    V = (mesh.vertices) * scale
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    result_crop = []
    colors_rgb = mesh.visual.face_colors
    for brick in tile_set:
        inside, nearby_face = brick_inside(brick, mesh)
        if inside:
            nearby_color = colors_rgb[nearby_face]
            nearby_hex = RGB_to_Hex(nearby_color)
            new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
            result_crop.append(new_brick)
    return result_crop

if __name__ == "__main__":
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/bunny.obj")
    tile_path = os.path.join(os.path.dirname(__file__), "super_graph/['3004'] 5.ldr")
    tile_set = read_bricks_from_file(tile_path)
    mesh = trimesh.load(obj_path)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    scale = float(input("Enter scale of obj: "))
    start_time = time.time()
    result = crop_brick(mesh, tile_set, scale)
    print(f"resulting LEGO model has {len(result)} bricks")

    debugger = MyDebugger("test")
    _, filename=os.path.split(obj_path)
    filename = (filename.split("."))[0]
    _, tilename=os.path.split(tile_path)
    tilename = (tilename.split("."))[0]
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} s={scale} n={len(result)} {tilename} t={round(time.time() - start_time, 2)}.ldr"))