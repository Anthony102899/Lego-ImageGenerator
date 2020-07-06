import trimesh
import numpy as np
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file

obj_path = "./data/super_graph/bunny.obj"
tile_set = read_bricks_from_file("./data/super_graph/3004-ring7.ldr")

def brick_inside(brick, mesh):
    cpoints = brick.get_current_conn_points()
    cpoint_pos = list(map(lambda cp: list(cp.pos), cpoints))  # position of cpoints of *brick*
    cpoint_inside = mesh.contains(cpoint_pos)
    nearby_faces = trimesh.proximity.nearby_faces(mesh, cpoint_pos)
    if cpoint_inside.all():
        return True, [x[0] for x in nearby_faces]
    return False, [x[0] for x in nearby_faces]

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def get_face_color(mesh):
    colors_rgb = []
    face_num = len(mesh.faces)
    face_color = list(map(lambda f_color: list(trimesh.visual.color.to_rgba(f_color)), mesh.visual.face_colors))  # #faces * 4
    for face in range(face_num):
        colors_rgb.append(face_color[face])
    return colors_rgb

def crop_brick(mesh, tile_set, scale):
    V = (mesh.vertices) * scale
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    colors_rgb = get_face_color(mesh)
    result_crop = []
    for brick in tile_set:
        inside, nearby_face = brick_inside(brick, mesh)
        if inside:
            nearby_color = [colors_rgb[i] for i in nearby_face]
            color = np.average(nearby_color, axis=0)
            color_hex = RGB_to_Hex(color)
            #TODO
            result_crop.append(brick)
    return result_crop

if __name__ == "__main__":
    
    mesh = trimesh.load(obj_path)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    scale = float(input("Enter scale of obj: "))
    result = crop_brick(mesh, tile_set, scale)
    print(f"resulting LEGO model has {len(result)} bricks")