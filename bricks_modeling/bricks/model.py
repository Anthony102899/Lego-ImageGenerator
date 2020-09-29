from bricks_modeling.bricks.brick_group import BrickGroup
import numpy as np

# a model consists of a hierarchy of groups of bricks
class Model:
    def __init__(self, group_names):
        self.main_model_name = group_names[0]
        self.group_names = group_names
        self.groups = {name: BrickGroup(name) for name in group_names}

    def get_root_file(self):
        return self.groups[self.main_model_name]

    def is_empty(self):
        return len(self.groups) == 0

    def get_bricks(self):
        bricks = read_bricks_from_group(self.get_root_file(), np.identity(4, dtype=float), self.groups)
        return bricks

def read_bricks_from_group(file: BrickGroup, trans_matrix, all_files):
    bricks = []
    # read bricks of the current group
    bricks = bricks + file.get_transformed_bricks(trans_matrix)

    # read bricks of the sub groups
    for i in range(len(file.subgroups)):
        matrix_for_subgroup = file.trans_matrix_for_subgroups[i]
        new_trans_matrix = trans_matrix @ matrix_for_subgroup
        sub_file = all_files[file.subgroups[i]]
        bricks = bricks + read_bricks_from_group(sub_file, new_trans_matrix, all_files)

    return bricks




