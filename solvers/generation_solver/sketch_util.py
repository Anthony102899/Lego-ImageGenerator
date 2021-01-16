import os
import numpy as np
import math
import json
from shapely.geometry import Polygon, Point
import cv2
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from shapely.ops import unary_union
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.file_IO.model_reader import read_bricks_from_file

class Crop:            # sd of nodes
    def __init__(self, result_sd, result_color, base_count, filename, platename):
        self.result_sd = result_sd
        self.result_color = result_color
        self.base_count = base_count
        self.filename = filename
        self.platename = platename

def center_crop(img, scale):
    width, height = img.shape[1], img.shape[0]
    dim = (height / scale, width / scale)
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2) 
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

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

def move_layer(brickset, layer_num):
    current_y = (brickset[0].get_translation())[1]
    new_set = brickset.copy()
    dis = 8 * layer_num - current_y
    [b.translate([0, dis, 0]) for b in new_set]
    return new_set

def hex_to_rgb(hexx):
    value = hexx.lstrip('0x2')
    lv = len(value)
    if lv == 0:
        return np.array([0,0,0])
    rgb = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return np.array(rgb)

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

# get a new brick with the input color
def color_brick(brick, rgb_color):
    color_hex = RGB_to_Hex(rgb_color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color_hex)
    return new_brick

# return a list of rgb colors covered by brick *rgbs*
def get_cover_rgb(brick, img, base_int):
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
                    if channel == 4 and img[y, x][0] == 0:
                        return []
                    # not transparent
                    else:
                        rgbs.append(rgb_color)
                except:
                    continue
    return rgbs

def load_data(brick_database=["regular_plate.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling", "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    return data

def get_weight():
    data = load_data()
    area = {}
    for brick in data:
        if "weight" in brick.keys():
            area.update({brick["id"]: brick["weight"]})
        else:
            area.update({brick["id"]: 1})
    return area

def get_area():
    data = load_data()
    area = {}
    for brick in data:
        if "area" in brick.keys():
            area.update({brick["id"]: brick["area"]})
    return area

# return difference between input and brickset (solution) (without base)
def calculate_v(brick_set,img, base_int, polygon_ls):
    dif_sum = [0, 0, 0]
    for i in range(len(brick_set)):
        brick = brick_set[i]
        polygon = polygon_ls[i]
        brick_color = hex_to_rgb(brick.color)
        mini, minj, maxi, maxj = polygon.bounds
        dif = [0, 0, 0]
        for x in range(math.floor(mini), math.ceil(maxi) + 1):
            for y in range(math.floor(minj), math.ceil(maxj) + 1):
                point = Point(x, y)
                if polygon.contains(point):
                    try:
                        img_color = (img[y, x])[::-1]
                        dif = [dif[i] + abs(brick_color[i] - img_color[i]) * 1e-6 for i in range(3)]
                    except:
                        continue
        dif_sum = [round(dif_sum[i] + dif[i], 3) for i in range(3)]
    return - np.sum(dif_sum)

# *brick* is the lower one
def calculate_overlap_v(brick, brick2, img, base_int):
    polygon1 = proj_bbox(brick)
    polygon2 = proj_bbox(brick2)
    dif_polygon = polygon1.difference(polygon2)
    return calculate_v([brick], img, base_int, [dif_polygon])

# return an integer in [0,1]
def cal_border(brickset, base_int):
    standard = base_int * 4 - 4
    maxx = base_int * 20 - 10
    count = 0
    for brick in brickset:
        cpoints = brick.get_current_conn_points()
        cpoints_pos = [[cp.pos[0], cp.pos[2]] for cp in cpoints]
        for z in range(10, base_int * 20 -9, 10):
            if [10, z] in cpoints_pos or [maxx, z] in cpoints_pos:
                count += 1
            if z < 20 or z > maxx - 10:
                continue
            if [z, 10] in cpoints_pos or [z, maxx] in cpoints_pos:
                count += 1
    return count / standard


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "super_graph/images/pepsi.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    image = center_crop(img, 2)
    
    cv2.imwrite('./solvers/generation_solver/super_graph/test.png', image)