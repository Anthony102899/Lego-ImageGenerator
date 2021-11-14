import os
import numpy as np
import math
import json
import sys
from shapely.geometry import Polygon, Point
import cv2
from bricks_modeling.bricks.brickinstance import BrickInstance, get_corner_pos
from shapely.ops import unary_union
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from util.debugger import MyDebugger

# get a new brick with the input color
def color_brick(brick, color, rgb=True):
    if rgb:
        color = RGB_to_Hex(color)
    new_brick = BrickInstance(brick.template, brick.trans_matrix, color)
    return new_brick

def count_base_number(plate_set):
    base_count = 0
    for i in range(100):
        if plate_set[i].color == 15:
            base_count += 1
        else:
            break
    return base_count

def get_area():
    data = load_data()
    area = {}
    for brick in data:
        if "area" in brick.keys():
            area.update({brick["id"]: brick["area"]})
    return area

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
                    bgra = img[y, x]
                    rgb_color = (bgra[:3])[::-1]
                    if channel == 4 and bgra[3] == 0:
                        return []
                    # not transparent
                    else:
                        rgbs.append(rgb_color)
                except:
                    continue
    return rgbs

def get_weight():
    data = load_data()
    area = {}
    for brick in data:
        if "weight" in brick.keys():
            area.update({brick["id"]: brick["weight"]})
        else:
            area.update({brick["id"]: 1})
    return area

def hex_to_rgb(value):
    if len(value) < 6:
        return np.array([0, 0, 0])
    if len(value) > 6:
        value = value.lstrip('0x2')
    lv = len(value)
    rgb = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return np.array(rgb)

def load_data(brick_database=["regular_plate.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling", "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    return data

def move_brickset(brickset, rgb_color, x, z, ldr_color):
    ldr_color = nearest_color(rgb_color, ldr_color)
    new_set = [color_brick(brick, ldr_color, rgb=False) for brick in brickset]
    [brick.translate([x, 0, z]) for brick in new_set]
    return new_set

def move_layer(brickset, layer_num):
    new_set = brickset.copy()
    goal = 8 * layer_num
    for brick in new_set:
        current_y = (brick.get_translation())[1]
        if current_y == goal:
            continue
        brick.translate([0, goal - current_y, 0])
    return new_set

# return an integer
def nearest_color(rgb, ldr_color):
    minn = sys.maxsize
    result = -1
    for key in ldr_color:
        rgb_key = hex_to_rgb(key["hex"].lstrip('#'))
        dif = rgb_key - rgb
        dif = np.linalg.norm(dif)
        if dif < minn:
            minn = dif
            result = key["LDR_code"]
    return result
        
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

# return a dictionary
def read_ldr_color():
    k = -1
    ldr_color = []
    for line in open(os.path.join(os.path.dirname(__file__), "StudioColorDefinition.txt")):
        k += 1
        if k > 0:
            line = (line[:-1].split("\t"))
            ldr_code = (int)(line[2])
            if ldr_code < 30 and not (ldr_code == 16 or ldr_code == 24):
                hex_value = line[8]
                ldr_color.append({"LDR_code": ldr_code, "hex": hex_value})
    return ldr_color

def RGB_to_Hex(rgb):
    color = '0x2'
    for i in rgb[:3]:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def rotate_image(img, angle):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def scale_image(img, scale):
    width, height = img.shape[1], img.shape[0]
    dim = (height / scale, width / scale)
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2) 
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

def translate_image(img, width_dis, height_dis):
    height, width = img.shape[:2] 
    T = np.float32([[1, 0, width_dis], [0, 1, height_dis]]) 
    img_translation = cv2.warpAffine(img, T, (width, height)) 
    return img_translation


if __name__ == "__main__":
    print(nearest_color("FF0008"))