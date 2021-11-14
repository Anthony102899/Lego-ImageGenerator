import numpy as np
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.bricks.brick_step import BrickStep
import copy

# to represent a group of bricks, which main contain a subgroup of bricks
class BrickGroup:
    def __init__(self, name):
        self.name = name
        self.brick_steps = [BrickStep()]

    def add_brick(self, line_content, brick_templates, template_ids, read_fake_brick=False):
        self.brick_steps[-1].add_a_brick(line_content, brick_templates, template_ids, read_fake_brick=True)

    def add_a_subgroup(self, line_content):
        self.brick_steps[-1].add_a_subgroup(line_content)

    def add_a_step(self):
        self.brick_steps.append(BrickStep())

    def get_all_subgroups(self):
        subgroups = []
        for brick_step in self.brick_steps:
            for i in range(len(brick_step.subgroup_names)):
                subgroups.append((brick_step.subgroup_names[i], brick_step.subgroups_colors[i], brick_step.subgroups_transformation[i]))
        return subgroups

    def get_transformed_bricks(self, trans_matrix):
        bricks = []
        for brick_step in self.brick_steps:
            for bricktemplate in brick_step.bricks:
                brick = copy.deepcopy(bricktemplate)
                brick.trans_matrix = trans_matrix @ brick.trans_matrix
                bricks.append(brick)
        return bricks

