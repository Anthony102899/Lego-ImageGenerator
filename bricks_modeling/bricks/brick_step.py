from typing import List
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.bricktemplate import BrickTemplate
import numpy as np

class BrickStep:
    def __init__(self):
        self.bricks = []
        self.subgroup_names = []
        self.subgroups_transformation = []
        self.subgroups_colors = []

    def add_a_subgroup(self, line_content):
        trans_matrix_for_internal_file = np.identity(4, dtype=float)
        translate = np.zeros((3, 1))
        for j in range(3):
            translate[j] = float(line_content[j + 2])

        rotation = np.identity(3, dtype=float)
        for j in range(9):
            rotation[j // 3][j % 3] = float(line_content[j + 5])

        trans_matrix_for_internal_file[:3, 3:4] = translate
        trans_matrix_for_internal_file[:3, :3] = rotation

        self.subgroups_colors.append(int(line_content[1]))
        self.subgroup_names.append(" ".join(line_content[14:]).lower())
        self.subgroups_transformation.append(trans_matrix_for_internal_file)

    def add_a_brick(
            self, line_content, brick_templates, template_ids, read_fake_brick=False
    ):
        brick_id = line_content[-1][0:-4]
        # processing brick color
        if line_content[1].isdigit():
            color = int(line_content[1])
        else:
            color = line_content[1]

        translate = np.zeros((3, 1))
        for j in range(3):
            translate[j] = float(line_content[j + 2])

        rotation = np.identity(3, dtype=float)
        for j in range(9):
            rotation[j // 3][j % 3] = float(line_content[j + 5])

        if brick_id in template_ids:
            brick_idx = template_ids.index(brick_id)
            brickInstance = BrickInstance(brick_templates[brick_idx], np.identity(4, dtype=float), color)
        elif read_fake_brick:
            brickInstance = BrickInstance(BrickTemplate([], brick_id), np.identity(4, dtype=float), color)
        else:
            print(f"cannot find {brick_id} in database, and do not allow virtual brick reading!")
            exit(7)

        brickInstance.rotate(rotation)
        brickInstance.translate(translate)
        self.bricks.append(brickInstance)
