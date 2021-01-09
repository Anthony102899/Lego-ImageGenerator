import numpy as np
import os
import math
import cv2
import pickle5 as pickle
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from solvers.generation_solver.crop_model import RGB_to_Hex
from multiprocessing import Pool
from functools import partial

class Crop:            # sd of nodes
    def __init__(self, result_crop, base_count, filename, platename):
        self.result_crop = result_crop
        self.base_count = base_count
        self.filename = filename
        self.platename = platename

# return a polygon obj 
def proj_bbox(brick:BrickInstance):
    bbox_corner = np.array(get_corner_pos(brick, four_point=True))
    bbox_corner = [[coord[0], coord[2]] for coord in bbox_corner]
    polygon_ls = []
    for i in range(0, len(bbox_corner), 4):
        polygon = Polygon(bbox_corner[i:i+4])
        polygon_ls.append(polygon)
    polygon = unary_union(polygon_ls)
    return polygon

# return a list of rgb colors covered by brick
def get_cover_rgb(img, brick, base_int):
    polygon = proj_bbox(brick)
    mini, minj, maxi, maxj = polygon.bounds
    rgbs = []
    channel = len(img[0][0])
    for x in range(math.floor(mini), math.ceil(maxi) + 1):
        for y in range(math.floor(minj), math.ceil(maxj) + 1):
            if x < 0 or y < 0 or x > base_int * 20 or y > base_int * 20:
                return []
            point = Point(x, y)
            if polygon.contains(point):
                try:
                    rgb_color = (img[y, x][:3])[::-1]
                    if channel == 4 and (img[y, x][0] == 0 or sum(np.array(rgb_color)) == 0):
                        continue
                    # not transparent
                    else:
                        rgbs.append(rgb_color)
                except:
                    continue
    return rgbs

# get a new brick with the input color
def colored_brick(brick, color):
    color_hex = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color_hex)
    return new_brick

def color_brick_ls(brick, img, base_int):
    rgbs = get_cover_rgb(img, brick, base_int)
    if len(rgbs) == 0:
        return
    color = np.average(rgbs, axis = 0)
    return colored_brick(brick, color), rgbs

def sketch_from_layout(img, plate_set, base_int):
    with Pool(20) as p:
        result_crop = p.map(partial(color_brick_ls, img=img, base_int=base_int), plate_set)
    return [i for i in result_crop if i]

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/images/heart.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    plate_path = "super_graph/for sketch/" + input("Enter file name in sketch folder: ")
    plate_path = os.path.join(os.path.dirname(__file__), plate_path)
    plate_set = read_bricks_from_file(plate_path)
    
    base_count = 0
    for i in range(10):
        if plate_set[i].color == 15:
            base_count += 1
        else:
            break
    plate_base = plate_set[:base_count]
    plate_set = plate_set[base_count:]
    print("#bricks in plate: ", len(plate_set))

    _, filename = os.path.split(img_path)
    filename = (filename.split("."))[0]
    _, platename = os.path.split(plate_path)
    platename = (((platename.split("."))[0]).split("n="))[0]

    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in plate_base])
    base_int = int(math.sqrt(np.sum(cpoints)))

    # resize image to fit the brick
    img = cv2.resize(img, (base_int * 20 + 1, base_int * 20 + 1))

    result_crop = sketch_from_layout(img, plate_set, base_int)

    debugger = MyDebugger("sketch")
    result = plate_base + [i[0] for i in result_crop]
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} b={base_count} n={len(result)} {platename}.ldr"))

    crop_result = Crop([0.0001 for i in range(base_count)] + [float(round(np.sum(np.std(i[1], axis = 0)), 4) + 0.0001) for i in result_crop], base_count, filename, platename)
    pickle.dump(crop_result, open(os.path.join(os.path.dirname(__file__), f"connectivity/crop_{filename} b={base_count} n={len(result)} {platename}.pkl"), "wb"))