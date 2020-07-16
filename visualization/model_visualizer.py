import open3d as o3d

from bricks_modeling.connectivity_graph import ConnectivityGraph
from bricks_modeling.file_IO.model_converter import color_phraser
from bricks_modeling.file_IO.model_reader import read_bricks_from_file

if __name__ == "__main__":
    color_dict = color_phraser()
    ldr_path = "../data/full_models/42023.mpd"
    bricks = read_bricks_from_file(ldr_path)
    connect_graph = ConnectivityGraph(bricks)
    meshs = o3d.geometry.TriangleMesh()
    for brick in bricks:
        meshs += brick.get_mesh(color_dict)
    mesh,line_set =  connect_graph.get_mesh()
    meshs+=mesh
    o3d.visualization.draw_geometries([meshs, line_set])