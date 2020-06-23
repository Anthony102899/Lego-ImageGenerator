from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.connections.conn_type import ConnType
import numpy as np


if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/full_models/hinged_L.ldr")
    write_bricks_to_file(bricks, file_path=debugger.file_path("test.ldr"), debug=False)
    structure_graph = ConnectivityGraph(bricks)

    points = []
    points_on_brick = {i: [] for i in range(len(bricks))}

    #### sampling variables on contact points
    for edge in structure_graph.edges:
        if edge["type"] == ConnType.HOLE_PIN.name:
            bi = edge["node_indices"][0]
            bj = edge["node_indices"][1]
            contact_pt = edge["properties"]["contact_point"]
            contact_orient = edge["properties"]["contact_orient"]

            p0 = contact_pt
            p1 = contact_pt + 5 * contact_orient
            p2 = contact_pt - 5 * contact_orient

            points.append(contact_pt + 5 * contact_orient)
            points_on_brick[bi].append(len(points) - 1)
            points_on_brick[bj].append(len(points) - 1)

            points.append(contact_pt - 5 * contact_orient)
            points_on_brick[bi].append(len(points) - 1)
            points_on_brick[bj].append(len(points) - 1)

            print("pos", contact_pt, "orient", contact_orient)

    #### add additional sample points

    print(np.array(points))
    print(points_on_brick)
