
from solvers.generation_solver.crop_model import brick_inside, RGB_to_Hex, get_color
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.gurobi_solver import GurobiSolver
from bricks_modeling.bricks.brickinstance import BrickInstance
from solvers.generation_solver.gen_super_graph import get_volume
from util.debugger import MyDebugger
from multiprocessing import Pool
from functools import partial
import trimesh
import numpy as np
import time
import pickle5 as pickle


def check_brick1(brick, mesh, colors_rgb):
    inside, nearby_face = brick_inside(brick, mesh)
    if inside:
        nearby_color = colors_rgb[nearby_face]
        nearby_hex = RGB_to_Hex(nearby_color)
        new_brick = BrickInstance(brick.template, brick.trans_matrix, nearby_hex)
        return new_brick, 1
    return None, -1

def get_bricks(mesh, tile_set, scale):
    colors_rgb = get_color(mesh)
    V = (mesh.vertices) * scale
    mesh = trimesh.Trimesh(vertices=V, faces=mesh.faces)
    with Pool(20) as p:
        result = p.map(partial(check_brick1, mesh=mesh, colors_rgb=colors_rgb), tile_set)
    result = np.array(result)
    flag = result[:,1]
    result_crop = result[:,0]
    result_crop = [b for b in result_crop if b]
    return result_crop, flag

if __name__ == "__main__":
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/pokeball.ply")
    tile_path = os.path.join(os.path.dirname(__file__), "connectivity/['3004', '3062'] 3.pkl")
    tile = pickle.load(open(tile_path, "rb"))
    tile_set = tile.bricks
    print("#bricks in tile: ", len(tile_set))
    mesh = trimesh.load_mesh(obj_path)
    if not type(mesh) == trimesh.Trimesh:
        mesh = mesh.dump(True)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    debugger = MyDebugger("test")
    for scale in range (5, 6):
        #scale = float(input("Enter scale of obj: "))
        start_time = time.time()
        scale /= 10
        tile_set = tile.bricks
        result_crop, flag = get_bricks(mesh, tile_set, scale)
        print(flag[:10])
        print(len(result_crop))
        end_time = time.time()
        print(f"resulting LEGO model has {len(result_crop)} bricks")

        _, filename = os.path.split(obj_path)
        filename = (filename.split("."))[0]
        _, tilename = os.path.split(tile_path)
        tilename = ((tilename.split("."))[0]).split(" ")
        tilename = tilename[0] + tilename[1] + tilename[2]
        write_bricks_to_file(
            result_crop, 
            file_path=debugger.file_path(
                f"{filename} s={scale} n={len(result_crop)} {tilename} t={round(end_time - start_time, 2)}.ldr"))

        volume = get_volume()
        start_time = time.time()
        solver = GurobiSolver()
        results, time_used = solver.solve(nodes_num=len(tile_set),
                                        node_volume=[volume[b.template.id] for b in tile_set],
                                        edges=tile.overlap_edges,
                                        flag=flag)
        end_time = time.time()
        selected_bricks = []
        for i in range(len(tile.bricks)):
            if results[i] == 1:
                selected_bricks.append(tile.bricks[i])

        write_bricks_to_file(
            selected_bricks, file_path=debugger.file_path(f"selected n={len(selected_bricks)} t={round(end_time - start_time, 2)}.ldr"))

        print("done!")
