import trimesh
import numpy as np
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
    obj_path = "./data/super_graph/bunny.obj"
    tile_set = read_bricks_from_file("./data/super_graph/3004-ring7.ldr")
    mesh = trimesh.load(obj_path)
    scale = float(input("Enter scale of obj: "))
    result = crop_brick(mesh, tile_set, scale)
    print(f"resulting LEGO model has {len(result)} bricks")

    debugger = MyDebugger("test")
    write_bricks_to_file(result, file_path=debugger.file_path(f"lego-{scale}.ldr"))