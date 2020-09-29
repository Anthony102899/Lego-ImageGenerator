from bricks_modeling.bricks.brick_group import BrickGroup
import numpy as np

# a model consists of a hierarchy of groups of bricks
class Model:
    def __init__(self, group_names):
        self.main_model_name = group_names[0]
        self.group_names = group_names
        self.groups = {name: BrickGroup(name) for name in group_names}

    def get_root_file(self) -> BrickGroup:
        return self.groups[self.main_model_name]

    def is_empty(self):
        return len(self.groups) == 0

    def get_bricks(self):
        bricks = get_all_bricks_from_current_group(self.get_root_file(), np.identity(4, dtype=float), self.groups)
        return bricks

def get_all_bricks_from_current_group(current_brick_group: BrickGroup, trans_matrix, all_subgroups):
    bricks = []
    # read bricks of the current group
    bricks = bricks + current_brick_group.get_transformed_bricks(trans_matrix)

    # read bricks of the sub groups
    subgroups = current_brick_group.get_all_subgroups()
    for group in subgroups:
        subgroup_name = group[0]
        subgroup_trans_matrix = group[2]
        new_trans_matrix = trans_matrix @ subgroup_trans_matrix
        sub_file = all_subgroups[subgroup_name]
        bricks = bricks + get_all_bricks_from_current_group(sub_file, new_trans_matrix, all_subgroups)

    return bricks


