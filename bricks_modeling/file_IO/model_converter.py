from os import path
import open3d as o3d
import trimesh

from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.database.ldraw_colors import color_phraser
from util.debugger import MyDebugger

"""
    color phraser is to convert the color code in ldview to RGB in open3d
    
    ldr_to_obj is to convert a ldr model to a obj mesh model
    
"""


def ldr_to_obj(
    ldr_path,
    output_path=MyDebugger("test").file_path("test.obj"),
    open3d_vis=False,
    write_file=False
):
    color_dict = color_phraser()
    bricks = read_bricks_from_file(ldr_path)
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        meshs += brick.get_mesh(color_dict)

    if open3d_vis:
        o3d.visualization.draw_geometries([meshs])

    if write_file:
        o3d.io.write_triangle_mesh(output_path, meshs)

    return meshs

if __name__ == "__main__":
    ldr_to_obj("/Users/wuyifan/lego-solver/debug/truck.ldr", write_file=True)