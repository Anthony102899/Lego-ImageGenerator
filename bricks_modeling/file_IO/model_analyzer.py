import itertools
import numpy as np
from bricks.conn_type import compute_conn_type
from structure_graph import StructureGraph

def build_graph_from_bricks(bricks):
    edges = []
    for b_i,b_j in itertools.combinations(list(range(0,len(bricks))), 2):
        brick_i_conn_points = bricks[b_i].get_current_conn_points()
        brick_j_conn_points = bricks[b_j].get_current_conn_points()

        for m in range(0, len(brick_i_conn_points)):
            for n in range(0, len(brick_j_conn_points)):
                cpoint_m = brick_i_conn_points[m]
                cpoint_n = brick_j_conn_points[n]
                type = compute_conn_type(cpoint_m, cpoint_n)
                if type is not None:
                    print(cpoint_m.pos, cpoint_m.orient, cpoint_n.pos, cpoint_n.orient, type)
                    edges.append({"type":type.name, "node_indices":[b_i,b_j],
                                  "properties":{
                                      "contact_point_1" : bricks[b_i].template.c_points[m].pos,
                                      "contact_orient_1": bricks[b_i].template.c_points[m].orient,
                                      "contact_point_2" : bricks[b_j].template.c_points[n].pos,
                                      "contact_orient_2": bricks[b_j].template.c_points[n].orient
                                  }})

    return StructureGraph(bricks=bricks, edges=edges)

