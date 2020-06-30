from os import path
import open3d as o3d

from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger

"""
    color phraser is to convert the color code in ldview to RGB in open3d
    
    ldr_to_obj is to convert a ldr model to a obj mesh model
    
"""


def color_phraser(
    file_path=path.join(
        path.dirname(path.dirname(__file__)), "database", "ldraw", "LDConfig.ldr"
    )
):
    color_dict = {}

    f = open(file_path, "r")
    for line in f.readlines():
        line_content = line.rstrip().split()
        if len(line_content) < 8 or line_content[1] == "//":
            continue
        else:
            color_dict[line_content[4]] = [
                int(line_content[6][i : i + 2], 16) / 255 for i in (1, 3, 5)
            ]
            print(f"color {line_content[4]} is {color_dict[line_content[4]]}")

    return color_dict


def ldr_to_obj(
    ldr_path,
    obj_file_path=path.join(path.dirname(path.dirname(__file__)), "database", "obj"),
    output_path=MyDebugger("test").file_path("test.obj"),
    open3d_vis=False,
):
    color_dict = color_phraser()
    bricks = read_bricks_from_file(ldr_path)
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        mesh = o3d.io.read_triangle_mesh(
            f'{obj_file_path}\{brick.template.id + ".obj"}'
        )
        mesh.compute_vertex_normals()
        if str(brick.color) in color_dict.keys():
            mesh.paint_uniform_color(color_dict[str(brick.color)])
        else:
            print("warning, no such color in ldview, print red")
            mesh.paint_uniform_color([1, 0, 0])
        mesh.rotate(brick.get_rotation().tolist(), [0, 0, 0])
        mesh.translate([i / 2.5 for i in brick.get_translation().tolist()])
        meshs += mesh

    if open3d_vis:
        o3d.visualization.draw_geometries([meshs])

    o3d.io.write_triangle_mesh(output_path, meshs)

    return meshs


if __name__ == "__main__":
    ldr_to_obj("../../data/full_models/42023.mpd", open3d_vis=True)
