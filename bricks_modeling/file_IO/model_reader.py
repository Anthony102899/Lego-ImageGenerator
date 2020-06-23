import copy
import math

import numpy as np

from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.file_IO import model_writer
from util.debugger import MyDebugger

class File():
    def __init__(self):
        self.name = ""
        self.father_file = None
        self.internal_file = []
        self.bricks = []
        self.trans_matrix_for_internal_file = []


def read_a_brick(bricks, line_content, brick_templates, template_ids):
    brick_id = line_content[-1][0:-4]
    if brick_id in template_ids:
        # processing brick color
        color = int(line_content[1])

        # processing the transformation matrix
        brick_idx = template_ids.index(brick_id)
        trans_matrix = np.identity(4, dtype=float)

        new_translate = np.zeros((3, 1))
        for j in range(3):
            new_translate[j] = float(line_content[j + 2])

        new_rotation = np.identity(3, dtype=float)
        for j in range(9):
            new_rotation[j // 3][j % 3] = float(line_content[j + 5])

        brickInstance = BrickInstance(brick_templates[brick_idx], np.identity(4, dtype=float),
                                      color)
        brickInstance.rotate(new_rotation)
        brickInstance.translate(new_translate)
        bricks.append(brickInstance)


def read_files(file_path):
    f = open(file_path, "r")
    files = []

    for line in f.readlines():
        line_content = line.rstrip().split(" ")
        if len(line_content) < 3:
            continue
        if line_content[0] == "0" and line_content[1] == "FILE":
            file_name = ""

            for j in range(2, len(line_content)):
                file_name = file_name + line_content[j] + " "
                files.append(file_name)

    # print(f"now all files are {files}")

    return files


def read_graph_from_file(file_path):
    f = open(file_path, "r")

    brick_templates, template_ids = get_all_brick_templates()
    files_name = read_files(file_path)
    lines = f.readlines()
    Files = []
    i = 0
    while i < len(lines):
        line_content = lines[i].rstrip().split(" ")

        if len(line_content) < 2:
            i += 1
            continue

        if not(line_content[0] == "0" and line_content[1] == "FILE") and not(line_content[0] == "1" and len(line_content) == 15):
            i+=1
            continue

        if line_content[0] == "0" and line_content[1] == "FILE" or (line_content[0] == "1" and len(line_content) == 15):
            if line_content[0] == "0" and line_content[1] == "FILE":
                file_name = ""

                for j in range(2, len(line_content)):
                    file_name = file_name + line_content[j] + " "

                print(f"Notice a new file {file_name}")
                new_file = File()
                new_file.name = file_name
                Files.append(new_file)
            elif line_content[0] == "1":
                file_name = "main"
                print(f"Notice a new file {file_name}")
                new_file = File()
                new_file.name = file_name
                Files.append(new_file)
            i += 1
            while i < len(lines):
                line_content = lines[i].rstrip().split(" ")
                if len(line_content) < 3:
                    i += 1
                    continue

                if line_content[0] == "0" and line_content[1] == "FILE":
                    # print(f"a different File ")
                    break

                if line_content[0] == "1":
                    file_name = ""

                    for j in range(14, len(line_content)):
                        file_name = file_name + line_content[j] + " "

                    if file_name not in files_name:
                        read_a_brick(new_file.bricks, line_content, brick_templates, template_ids)
                        i += 1
                        continue
                    elif file_name in files_name:
                        print(f"Notice a internal file {file_name} for {new_file.name}")
                        new_file.internal_file.append(file_name)
                        trans_matrix_for_this = np.identity(4, dtype=float)
                        new_translate = np.zeros((3, 1))
                        for j in range(3):
                            new_translate[j] = float(line_content[j + 2])

                        new_rotation = np.identity(3, dtype=float)
                        for j in range(9):
                            new_rotation[j // 3][j % 3] = float(line_content[j + 5])

                        trans_matrix_for_this[:3, 3:4] = new_translate
                        trans_matrix_for_this[:3, :3] = new_rotation

                        new_file.trans_matrix_for_internal_file.append(trans_matrix_for_this)
                        i += 1
                        continue

                i += 1

        #i += 1

    return Files


def find_nodes(Files):
    nodes = []
    for file in Files:
        flag = 0
        for file2 in Files:
            for filename in file2.internal_file:
                if file.name == filename:
                    flag = 1
        if flag == 0:
            nodes.append(file.name)
    return nodes


def read_bricks_from_a_file(bricks, file, trans_matrix):
    for bricktemplate in file.bricks:
        brick = copy.deepcopy(bricktemplate)
        brick.rotate(trans_matrix[:3, :3])
        brick.trans_matrix[:3, 3:4] = np.dot(trans_matrix[:3, :3], brick.trans_matrix[:3, 3:4])
        brick.translate(trans_matrix[:3, 3:4])
        bricks.append(brick)


def find_file_by_name(files, name):
    for file in files:
        if file.name == name:
            return file

    # print("no such file name")
    return None


def read_file_from_startfile(bricks, file, trans_matrix, files):
    # print(f"read bricks from {file.name}")
    read_bricks_from_a_file(bricks, file, trans_matrix)
    if len(file.internal_file) == 0:
        print(f"no internal file for {file.name}")
        return 1
    else:
        print(f"{file.name} has {len(file.internal_file)} internal files")
        for i in range(len(file.internal_file)):
            # print(f"now handling{file.internal_file[i]}")
            internal_file = find_file_by_name(files, file.internal_file[i])
            # print(f"file's name {internal_file.name}")
            new_trans_matrix = np.identity(4, dtype=float)
            new_trans_matrix[:3, :3] = np.dot(trans_matrix[:3, :3], (file.trans_matrix_for_internal_file[i])[:3, :3])
            new_trans_matrix[:3, 3:4] = np.dot(trans_matrix[:3, :3],
                                               (file.trans_matrix_for_internal_file[i])[:3, 3:4]) + trans_matrix[:3,
                                                                                                    3:4]
            read_file_from_startfile(bricks, internal_file, new_trans_matrix, files)


def read_bricks_from_graph(bricks, files):
    nodes = find_nodes(files)
    '''print(nodes)
    for file_name in nodes:
        file = find_file_by_name(files, file_name)
        read_file_from_startfile(bricks, file, np.identity(4, dtype=float), files)'''
    file = find_file_by_name(files, nodes[0])
    read_file_from_startfile(bricks, file, np.identity(4, dtype=float), files)


def read_bricks_from_file(file_path):
    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    files = read_graph_from_file(file_path)
    read_bricks_from_graph(bricks, files)
    return bricks

