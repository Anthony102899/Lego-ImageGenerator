import numpy as np
from util.debugger import MyDebugger
import itertools
import json
import pickle
import time
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import open3d as o3d
import copy
from solvers.generation_solver.tile_graph import unique_brick_list
from bricks_modeling.connections.conn_type import compute_conn_type
from util.json_encoder import NumpyArrayEncoder
from pathos.multiprocessing import ProcessingPool as Pool
import os

"""
To use a graph to describe a LEGO structure
"""

class AdjacencyGraph:
    def __init__(self, bricks):
        self.bricks = bricks
        self.connect_edges = []
        self.overlap_edges = []
        self._remove_redudant_bricks()

        self.build_graph_from_bricks()

    def _remove_redudant_bricks(self):
        print("#tiles before filtring repeat:", len(self.bricks))
        unique_brick_list(self.bricks)
        print("#tiles after filtring repeat:", len(self.bricks))

    def build(self, b_i, b_j):
        if self.bricks[b_i].collide(self.bricks[b_j]):
            return (b_i, b_j), 1
        elif self.bricks[b_i].connect(self.bricks[b_j]):
            return (b_i, b_j), 0
        return None, -1

    def build_graph_from_bricks(self):
        it = np.array(list(itertools.combinations(list(range(0, len(self.bricks))), 2)))
        with Pool(10) as p:
            a = p.map(self.build, it[:,0], it[:,1])

        for x in a:
            if x[1] == 1:
                self.overlap_edges.extend([x[0]])
            elif x[1] == 0:
                self.connect_edges.extend([x[0]])

    def to_json(self):
        nodes = []
        ##### Start json building
        for i in range(len(self.bricks)):
            brick = self.bricks[i]
            nodes.append(
                {
                    "translation": brick.get_translation(),
                    "orientation": [
                        brick.trans_matrix[0, 0],
                        brick.trans_matrix[0, 1],
                        brick.trans_matrix[0, 2],
                        brick.trans_matrix[1, 0],
                        brick.trans_matrix[1, 1],
                        brick.trans_matrix[1, 2],
                        brick.trans_matrix[2, 0],
                        brick.trans_matrix[2, 1],
                        brick.trans_matrix[2, 2],
                    ],
                }
            )

        return json.dumps({"nodes": nodes, "edges": self.connect_edges}, cls=NumpyArrayEncoder)

    def show(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.9, 0.1, 0.1])

        points = [b.get_translation().tolist() for b in self.bricks]

        spheres = o3d.geometry.TriangleMesh()
        for b in self.bricks:
            spheres += copy.deepcopy(sphere).translate(b.get_translation().tolist())

        lines = [e["node_indices"] for e in self.connect_edges]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=20, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([mesh_frame, line_set, spheres])

if __name__ == "__main__":
    path = "./inputs/for sketch/['43722', '43723'] base=24.ldr"
    bricks = read_bricks_from_file(path)
    _, filename = os.path.split(path)
    filename = (filename.split("."))[0]
    start_time = time.time()
    structure_graph = AdjacencyGraph(bricks)
    #print(structure_graph.connect_edges)
    t = round(time.time() - start_time, 2)
    pickle.dump(structure_graph, open(os.path.join(os.path.dirname(__file__), f'connectivity/{filename}.pkl'), "wb"))
    print(f"Saved at {filename}.pkl in t={t}")