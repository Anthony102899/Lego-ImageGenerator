import copy
import math
import numpy as np

def read_all_subgroup_names(file_path):
    f = open(file_path, "r")
    group_names = []

    for line in f.readlines():
        line_content = line.rstrip().split(" ")
        if len(line_content) < 3:
            continue
        if line_content[0] == "1" and len(group_names) == 0: # if no declearation for the main model, manually add one
            group_names.append("main.ldr")
        if line_content[0] == "0" and line_content[1] == "FILE":
            file_name = " ".join(line_content[2:])
            group_names.append(file_name.lower())

    return group_names

def get_file_name(line_content):
    assert line_content[0] == "0" and line_content[1] == "FILE"
    file_name = " ".join(line_content[2:])
    return file_name

def get_brick_name(line_content):
    assert line_content[0] == "1"
    if len(line_content) < 15:
        # print(f"Cannot find brick or file name in {line_content}")
        return ""
    else:
        file_name = " ".join(line_content[14:])
        return file_name.lower()

# example of "file declaration": 0 FILE submodel.ldr
def is_file_declaration(line_content):
    return line_content

def is_file_name_annotation(line_content):
    return line_content[0] == "0" and line_content[1] == "FILE"

def is_step_annotation(line_content):
    return line_content[0] == "0" and line_content[1].lower() == "step"

def is_a_brick(line_content, files_name):
    return line_content[0] == "1" and get_brick_name(line_content) not in files_name

def is_brick_group(line_content, files_name):
    return line_content[0] == "1" and get_brick_name(line_content) in files_name

def to_ldr_format(color, trans_matrix, part_id):
    text = (
            f"1 {color} {trans_matrix[0][3]} {trans_matrix[1][3]} {trans_matrix[2][3]} "
                    + f"{trans_matrix[0][0]} {trans_matrix[0][1]} {trans_matrix[0][2]} "
                    + f"{trans_matrix[1][0]} {trans_matrix[1][1]} {trans_matrix[1][2]} "
                    + f"{trans_matrix[2][0]} {trans_matrix[2][1]} {trans_matrix[2][2]} "
                    + f"{part_id}"
    )
    return text