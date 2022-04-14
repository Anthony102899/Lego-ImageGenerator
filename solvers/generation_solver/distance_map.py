import codecs
import json
import os.path

import cv2
import numpy as np

class DistanceMap:
    """
    DistanceMap Class: Convert a picture into distance of json format.

    Author: DING Baizeng
    Last Modified: 2022.04.07 - Add comments
    """
    def __init__(self, img_path, base_int):
        self.img = self.process_img(img_path, base_int)
        self.img_name = img_path.split("/")[-1].split(".")[0]

    def process_img(self, img_path, base_int):
        """
        Read and resize the img
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (base_int * 20 + 1, base_int * 20 + 1))
        return img

    def generate_distance_map(self, base_int):
        """
        Generate the distance map in json file format
        """
        MAX_DIS = base_int * 20 + 2
        channel = len(self.img[0][0])
        map = np.full((base_int * 20 + 1, base_int * 20 + 1), np.inf)
        boundary_level = -1
        while True:
            changed = False
            for x in range(base_int * 20 + 1):
                for y in range(base_int * 20 + 1):
                    if map[y][x] == np.inf:
                        try:
                            bgra = self.img[y, x]
                            if channel == 4 and bgra[3] == 0:
                                # Out of boundary
                                map[y][x] = MAX_DIS
                                changed = True
                            else:
                                # On or inside the boundary
                                map[y][x] = 0
                                changed = True
                        except:
                            continue
                    else:
                        if map[y][x] == boundary_level:
                            for x_ in range(max(0, x-1), min(MAX_DIS-2, x+1)+1):
                                for y_ in range(max(0, y-1), min(MAX_DIS-2, y+1)+1):
                                    if x_ == x and y_ == y:
                                        continue
                                    if map[y_][x_] > boundary_level + 1:
                                        map[y_][x_] = boundary_level + 1
                                        changed = True
            if not changed:
                break
            boundary_level += 1
        self.dump_to_json(map)

    def dump_to_json(self, map):
        json_list = map.tolist()
        json_path = os.path.dirname(__file__) + f"/json/{self.img_name}.json"
        json.dump(json_list, codecs.open(json_path, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)

def display(map_path):
    """
    display json format distance map
    """
    map = np.array(json.load(open(map_path)))
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == 0:
                print(f"\033[31m{0}\33[0m", end='  ')
            elif map[i][j] <= 0:
                print(f"\033[33m{'$'}\033[0m", end='  ')
            else:
                print(f"#", end='  ')
        print("")


if __name__ == "__main__":
    """map_path = "/Users/walter/Documents/FYP/lego-solver/solvers/generation_solver/json/wechat_white.json"
    display(map_path)"""

    file_path = os.path.dirname(__file__) + "/new_inputs/LEGO_e/LEGO_e_white.png"
    distance_map = DistanceMap(file_path, base_int=24)
    distance_map.generate_distance_map(base_int=24)







