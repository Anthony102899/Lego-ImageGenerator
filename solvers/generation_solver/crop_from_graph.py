import os
from solvers.generation_solver.crop_model import brick_inside, RGB_to_Hex, get_color
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.minizinc_solver import MinizincSolver
from bricks_modeling.bricks.brickinstance import BrickInstance
from solvers.generation_solver.gen_super_graph import get_volume
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from util.debugger import MyDebugger
from multiprocessing import Pool
from functools import partial
import trimesh
import numpy as np
import time
import pickle5 as pickle


def check_brick(brick, mesh, colors_rgb):
    inside, nearby_face = brick_inside(brick, mesh)
    if inside:
        nearby_color = colors_rgb[nearby_face]
        nearby_hex = RGB_to_Hex(nearby_color)
        new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
        return new_brick, 1
    return brick, -10

def get_bricks(mesh, tile_set, scale):
    colors_rgb = get_color(mesh)
    V = (mesh.vertices) * scale
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    with Pool(20) as p:
        result = p.map(partial(check_brick, mesh=mesh, colors_rgb=colors_rgb), tile_set)
    result = np.array(result)
    flag = result[:,1]
    result_crop = result[:,0]
    return result_crop, flag

if __name__ == "__main__":
    model_file = "./solvers/generation_solver/solve_model.mzn"
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/pokeball.ply")

    tile_path = os.path.join(os.path.dirname(__file__), "connectivity/['3005', '4287'] 6 n=11209 t=80429.55.pkl")
    structure_graph = pickle.load(open(tile_path, "rb"))
    tile_set = structure_graph.bricks
    print("#bricks in tile: ", len(tile_set))

    mesh = trimesh.load_mesh(obj_path)
    if not type(mesh) == trimesh.Trimesh:
        mesh = mesh.dump(True)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    volume = get_volume()
    debugger = MyDebugger("test")

    scale = float(input("Enter scale of obj: "))
    start_time = time.time()
    result_crop, flag = get_bricks(mesh, tile_set, scale)
    structure_graph.bricks = result_crop
    end_time = time.time()
    print("\nCropping time = ", end_time - start_time)

    _, filename = os.path.split(obj_path)
    filename = (filename.split("."))[0]
    _, tilename = os.path.split(tile_path)
    tilename = ((tilename.split("."))[0]).split(" ")
    tilename = tilename[0] + tilename[1]

    start_time = time.time()
    solver = MinizincSolver(model_file, "gurobi")
    results, time_used = solver.solve(structure_graph=structure_graph,
                                      node_volume=[volume[b.template.id] for b in structure_graph.bricks],
                                      flag=flag)
    end_time = time.time()
    selected_bricks = []
    for i in range(len(structure_graph.bricks)):
        if results[i] == 1:
            selected_bricks.append(structure_graph.bricks[i])
    print(f"Resulting LEGO model has {len(selected_bricks)} bricks")

    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected {filename} {tilename} s={scale} n={len(selected_bricks)} t={round(end_time - start_time, 2)}.ldr"))
    print("done!")
