import copy
import math

import numpy as np

from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.file_IO import model_writer
from util.debugger import MyDebugger


class File():
    def __init__(self, name):
        self.name = name
        self.father_file = None
        self.internal_file = []
        self.bricks = []
        self.trans_matrix_for_internal_file = []

    def add_an_internal_file(self, line_content):
        trans_matrix_for_internal_file = np.identity(4, dtype=float)
        translate = np.zeros((3, 1))
        for j in range(3):
            translate[j] = float(line_content[j + 2])

        rotation = np.identity(3, dtype=float)
        for j in range(9):
            rotation[j // 3][j % 3] = float(line_content[j + 5])

        trans_matrix_for_internal_file[:3, 3:4] = translate
        trans_matrix_for_internal_file[:3, :3] = rotation

        self.internal_file.append(get_file_or_brick_name(line_content))
        self.trans_matrix_for_internal_file.append(trans_matrix_for_internal_file)

    def read_a_brick(self,line_content, brick_templates, template_ids):
        brick_id = line_content[-1][0:-4]
        if brick_id in template_ids:
            # processing brick color
            color = int(line_content[1])

            # processing the transformation matrix
            brick_idx = template_ids.index(brick_id)
            translate = np.zeros((3, 1))
            for j in range(3):
                translate[j] = float(line_content[j + 2])

            rotation = np.identity(3, dtype=float)
            for j in range(9):
                rotation[j // 3][j % 3] = float(line_content[j + 5])

            brickInstance = BrickInstance(brick_templates[brick_idx], np.identity(4, dtype=float),
                                          color)
            brickInstance.rotate(rotation)
            brickInstance.translate(translate)
            self.bricks.append(brickInstance)
        else:
            print(f"cannot find {brick_id} in database")


class File_Tree():
    def __init__(self):
        self.files = []

    def find_root(self):
        if len(self.files) == 0:
            print("Error, no nodes")
        return self.files[0]

    def find_file_by_name(self, name):
        for file in self.files:
            if file.name == name:
                return file
        print("Error, no such file name")
        return None

def read_files_name(file_path):
    f = open(file_path, "r")
    files_name = []

    for line in f.readlines():
        line_content = line.rstrip().split(" ")
        if len(line_content) < 3:
            continue
        if line_content[0] == "0" and line_content[1] == "FILE":
            file_name = ""

            for j in range(2, len(line_content)):
                file_name = file_name + line_content[j] + " "
                files_name.append(file_name)

    # print(f"now all files are {files}")

    return files_name

def get_file_or_brick_name(line_content):

    if line_content[0] == "1": # must be a brick declaration or parts quotation

        if len(line_content) < 15:

            print(f"Cannot find brick or file name in {line_content}")
            return ""
        else:
            file_name = ""
            for j in range(14, len(line_content)):
                file_name = file_name + line_content[j] + " "
            return file_name

    if line_content[0] == "0" and line_content[1] == "FILE": # must be a parts declaration
        if len(line_content) < 3:
            print(f"Cannot find brick or file name in {line_content}")
            return ""
        else:
            file_name = ""
            for j in range(2, len(line_content)):
                file_name = file_name + line_content[j] + " "
            return file_name


def is_brick_declaration(line_content, files_name):
    return line_content[0] == "1" and get_file_or_brick_name(line_content) not in files_name

def is_parts_declaration(line_content, files_name):
    return line_content[0] == "0" and get_file_or_brick_name(line_content) in files_name

def is_parts_quotation(line_content, files_name):
    return line_content[0] == "1" and get_file_or_brick_name(line_content) in files_name

def read_tree_from_file(file_path):
    f = open(file_path, "r")
    brick_templates, template_ids = get_all_brick_templates()
    files_name = read_files_name(file_path)
    lines = f.readlines()
    file_tree = File_Tree()
    current_file = None
    for line in lines:
        line_content = line.rstrip().split(" ")
        if len(line_content) < 2:
            #print(f"too small length for {line}, pass")
            continue
        elif is_parts_declaration(line_content, files_name):
            file_name = get_file_or_brick_name(line_content)
            print(f"Notice a new file {file_name}")
            new_file = File(file_name)
            file_tree.files.append(new_file)
            current_file = new_file
        elif is_brick_declaration(line_content, files_name):
            if current_file == None:
                print(f"Notice there is no file but a brick delaration happens, so create a file named main")
                new_file = File("main")
                file_tree.files.append(new_file)
                current_file = new_file
            brick_name = get_file_or_brick_name(line_content)
            print(f"Notice a brick:{brick_name} for file:{current_file.name}")
            current_file.read_a_brick(line_content,brick_templates,template_ids)
        elif is_parts_quotation(line_content,files_name):
            internal_file_name = get_file_or_brick_name(line_content)
            print(f"Notice a internal file{internal_file_name} for current file :{current_file.name}")
            current_file.add_an_internal_file(line_content)
        else:
            print(f"unknown condition for line:{line}")
    return file_tree


def find_roots(Files):
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


def read_file_from_rootfile(bricks, file, trans_matrix, files):
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
            read_file_from_rootfile(bricks, internal_file, new_trans_matrix, files)


def read_bricks_from_graph(bricks, file_tree):
    #nodes = find_roots(files)
    file = file_tree.find_root()
    read_file_from_rootfile(bricks, file, np.identity(4, dtype=float), file_tree.files)


def read_bricks_from_file(file_path):
    bricks = []
    file_tree = read_tree_from_file(file_path)
    read_bricks_from_graph(bricks, file_tree)
    return bricks


#if __name__ == "__main__":
    #bricks = read_bricks_from_file("../../data/full_models/miniheads/standard.mpd")
    #debugger = MyDebugger("test")
    #model_writer.write_bricks_to_file(bricks, debugger.file_path("model.ldr"))





