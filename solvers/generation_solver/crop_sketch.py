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
from solvers.generation_solver.draw_bbox import write_bricks_w_bbox
from multiprocessing import Pool
from functools import partial

class Crop:
    def __init__(self, result_crop, base_count, filename, platename):
        self.result_crop = result_crop
        self.base_count = base_count
        self.filename = filename
        self.platename = platename

# output the pixels inside a brick
def output_pixel(brick, img, basename):
    polygon = proj_bbox(brick)
    dim = basename * 20 + 1
    mini, minj, maxi, maxj = polygon.bounds
    output_img = np.zeros([dim, dim,3])
    for x in range(math.floor(mini), math.ceil(maxi) + 1):
        for y in range(math.floor(minj), math.ceil(maxj) + 1):
            point = Point(x, y)
            if polygon.contains(point):
                try:
                    output_img[y][x] = img[y, x]
                except:
                    continue
    cv2.imwrite('./solvers/generation_solver/super_graph/brick_img.jpg', output_img)

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
def get_cover_rgb(img, brick, basename):
    polygon = proj_bbox(brick)
    mini, minj, maxi, maxj = polygon.bounds
    rgbs = []
    for x in range(math.floor(mini), math.ceil(maxi) + 1):
        for y in range(math.floor(minj), math.ceil(maxj) + 1):
            if x < 0 or y < 0 or x > basename * 20 or y > basename * 20:
                return []
            point = Point(x, y)
            if polygon.contains(point):
                try:
                    rgbs.append((img[y, x])[::-1])
                except:
                    continue
    return rgbs

# get a new brick with the nearest color
def check_sketch(brick, img, basename):
    rgbs = get_cover_rgb(img, brick, basename) # the nearest color
    if len(rgbs) == 0:
        return
    color = np.average(rgbs, axis = 0)
    color_hex = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color_hex)
    return new_brick, rgbs

def get_sketch(img, plate_set, basename):
    with Pool(20) as p:
        result_crop = p.map(partial(check_sketch, img=img, basename=basename), plate_set)
    return [i for i in result_crop if i]

def change_color_to_gray(brick, gray):
    color = np.ones(3) * gray
    color_hex = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color_hex)
    return new_brick

# return a list of bricks with gray scale color representing its sd
def sd_as_gray(brickset, img, basename):
    sd_ls = [round(np.sum(np.std(get_cover_rgb(img, brick, basename), axis = 0))) for brick in brickset]
    maxx = np.max(sd_ls)
    gray_ls = 255 - sd_ls / maxx * 255
    result = [change_color_to_gray(plate_set[i], gray_ls[i]) for i in range(len(plate_set))]
    return result

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/pepsi.png")
    img = cv2.imread(img_path)
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
    platename = (((platename.split("."))[0]).split("base="))[0]

    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in plate_base])
    basename = int(math.sqrt(np.sum(cpoints)))

    # resize image to fit the brick
    img = cv2.resize(img, (basename * 20 + 1, basename * 20 + 1))
    result_crop = get_sketch(img, plate_set, basename)
    
    debugger = MyDebugger("sketch")
    result = plate_base + [i[0] for i in result_crop]
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} b={base_count} n={len(result)} {platename}.ldr"))

    crop_result = Crop(result_crop, base_count, filename, platename)
    pickle.dump(crop_result, open(os.path.join(os.path.dirname(__file__), f"connectivity/crop_{filename} b={base_count} n={len(result)} {platename}.pkl"), "wb"))