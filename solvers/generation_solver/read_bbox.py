import os
import sys
ROOT_DIR = os.path.abspath('/Users/wuyifan/lego-solver')
sys.path.append(ROOT_DIR)
from bricks_modeling.bricks.brickinstance import BrickInstance, get_concave
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.connections.conn_type import compute_conn_type
import util.cuboid_geometry as cu_geo
import numpy as np
import itertools as iter

collider_path = "/Applications/Studio 2.0/ldraw/collider"

# {"Origin": self.pos, "Rotation": self._get_rotation_matrix(), "Dimension": dimension}
def read_bbox(brick):
    bbox = []
    brick_id = brick.template.id
    brick_rot = brick.get_rotation()
    brick_trans = brick.get_translation()
    #print(brick.template.id)
    #print("brick rot = \n", brick_rot)
    #print("brick trans = ", brick_trans,"\n")
    for line in open(os.path.join(collider_path, f"{brick_id}.col")):
        line = (line.split(" "))[:17]
        line = [float(x) for x in line]
        init_orient = (np.array(line[2:11])).reshape((3,3))
        #print("init_orient =\n", init_orient)
        init_origin = np.array(line[11:14])
        #print("init_origin = ", init_origin)
        init_dim = init_orient @ np.array(line[14:17])  # in (x,y,z) format
        #print("init_size = ", init_dim)

        origin = brick_rot @ init_origin + brick_trans
        #print("\norigin =\n", origin)
        rotation = brick_rot @ init_orient
        #print("rotation =\n", rotation)
        bbox.append({"Origin": origin, "Rotation": rotation, "Dimension": init_dim})
    return bbox

def tmp(brick1, brick2):
    bbox1 = read_bbox(brick1)
    bbox2 = read_bbox(brick2)
    #print(len(bbox1))
    concave = get_concave()
    concave_connect = 0
    for bb1, bb2 in iter.product(bbox1, bbox2):
        if cu_geo.cub_collision_detect(bb1, bb2):
            """
            if not compute_conn_type(p_self, p_other) == None:
                if bricks[1].template.id in concave or bricks[0].template.id in concave:
                    concave_connect = 1
                    continue
                print(0)
            """
            return 1
    return -1

if __name__ == "__main__":
    bricks = (read_bricks_from_file("./debug/test.ldr"))
    
    for i in range(len(bricks)):
        for j in range(len(bricks)):
            #print(f"{i}=={j}: ",bricks[i] == bricks[j])
            if not i==j and i > j:
                print(f"{i} collide with {j}: ", tmp(bricks[i], bricks[j]),"\n")
    """ TODO: check connection """