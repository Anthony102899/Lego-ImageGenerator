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

def check_brick(brick, mesh, colors_rgb):
    inside, nearby_face = brick_inside(brick, mesh)
    if inside:
        nearby_color = colors_rgb[nearby_face]
        nearby_hex = RGB_to_Hex(nearby_color)
        new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
        return new_brick
    return

def crop_brick(mesh, tile_set, scale):
    V = (mesh.vertices) * scale
    if type(mesh.visual) == trimesh.visual.ColorVisuals:
        colors_rgb = mesh.visual.face_colors
    else:
        colors_rgb = mesh.visual.to_color()
        if colors_rgb.defined:
            colors_rgb = colors_rgb.face_colors
        else:
            print("No valid color, using black!")
            colors_rgb = np.c_[np.zeros((len(mesh.faces),3)), np.array([255 for face in mesh.faces])]
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    """
    result_crop = []
    for brick in tile_set:
        inside, nearby_face = brick_inside(brick, mesh)
        if inside:
            nearby_color = colors_rgb[nearby_face]
            nearby_hex = RGB_to_Hex(nearby_color)
            new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
            result_crop.append(new_brick)
    """
    with Pool(20) as p:
        result_crop = p.map(partial(check_brick, mesh=mesh, colors_rgb=colors_rgb), tile_set)
    result_crop = [b for b in result_crop if b]
    return result_crop

if __name__ == "__main__":
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/pokeball.ply")
    tile_path = os.path.join(os.path.dirname(__file__), "super_graph/['3004', '4287'] 6 n=16369 t=89394.8.ldr")
    tile_set = read_bricks_from_file(tile_path)
    print("#bricks in tile: ", len(tile_set))
    mesh = trimesh.load_mesh(obj_path)
    if not type(mesh) == trimesh.Trimesh:
        mesh = mesh.dump(True)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    debugger = MyDebugger("test")
    for scale in range (10, 15):
        #scale = float(input("Enter scale of obj: "))
        start_time = time.time()
        scale /= 10
        result = crop_brick(mesh, tile_set, scale)
        end_time = time.time()
        print(f"resulting LEGO model has {len(result)} bricks")

        _, filename=os.path.split(obj_path)
        filename = (filename.split("."))[0]
        _, tilename=os.path.split(tile_path)
        tilename = ((tilename.split("."))[0]).split(" ")
        tilename = tilename[0] + tilename[1] + tilename[2]
        write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} s={scale} n={len(result)} {tilename} t={round(end_time - start_time, 2)}.ldr"))