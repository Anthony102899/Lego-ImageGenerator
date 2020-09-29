import numpy as np
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.bricks.bricktemplate import BrickTemplate
import copy

# to represent a group of bricks, which main contain a subgroup of bricks
class BrickGroup:
    def __init__(self, name):
        self.name = name
        self.bricks = []
        self.brick_steps = []
        self.subgroups = []
        self.trans_matrix_for_subgroups = []


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

        self.subgroups.append(" ".join(line_content[14:]).lower())
        self.trans_matrix_for_subgroups.append(trans_matrix_for_internal_file)

    def read_a_brick(
        self, line_content, brick_templates, template_ids, read_fake_brick=False
    ):
        brick_id = line_content[-1][0:-4]
        # processing brick color
        if line_content[1].isdigit():
            color = int(line_content[1])
        else: color = line_content[1]

        translate = np.zeros((3, 1))
        for j in range(3):
            translate[j] = float(line_content[j + 2])

        rotation = np.identity(3, dtype=float)
        for j in range(9):
            rotation[j // 3][j % 3] = float(line_content[j + 5])

        if brick_id in template_ids:
            # processing the transformation matrix
            brick_idx = template_ids.index(brick_id)

            brickInstance = BrickInstance(
                brick_templates[brick_idx], np.identity(4, dtype=float), color
            )
            brickInstance.rotate(rotation)
            brickInstance.translate(translate)
            self.bricks.append(brickInstance)
        elif read_fake_brick:
            brickInstance = BrickInstance(
                BrickTemplate([], brick_id), np.identity(4, dtype=float), color
            )
            brickInstance.rotate(rotation)
            brickInstance.translate(translate)
            self.bricks.append(brickInstance)
        else:
            print(f"cannot find {brick_id} in database, and do not allow virtual brick reading!")

    def get_transformed_bricks(self, trans_matrix):
        bricks = []
        for bricktemplate in self.bricks:
            brick = copy.deepcopy(bricktemplate)
            brick.trans_matrix = trans_matrix @ brick.trans_matrix
            bricks.append(brick)

        return bricks
