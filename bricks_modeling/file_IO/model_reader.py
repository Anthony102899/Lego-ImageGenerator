import numpy as np

from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance


def read_bricks_from_file(file_path):
    f = open(file_path, "r")

    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    transfrom_for_subs = {}
    transfrom_for_subs["origin"] = np.identity(4, dtype=float)
    subTurn = "origin"

    for line in f.readlines():
        line_content = line.rstrip().split(" ")

        """Detect The following lines are for another subparts"""
        if len(line_content)>1 and line_content[0] == "0" and "Sub" in line_content[1]:
            subTurn = line_content[1] + ".ldr"
            print(f"Notice This is a subgraph for {line_content[1]}:")
            continue

        """Detect new delcared subparts"""
        if line_content[0] == "1":
            if "Sub" in line_content[-1]:
                print(f"Notice a declared subgraph for {line_content[-1]}")

                new_trans_matrix = np.identity(4, dtype=float)
                for i in range(3):
                    new_trans_matrix[i][3] = float(line_content[i + 2])
                for i in range(9):
                    new_trans_matrix[i // 3][i % 3] = float(line_content[i + 5])

                this_trans_matrix = np.identity(4, dtype=float)
                this_trans_matrix[:3, :3] = np.dot(
                    transfrom_for_subs[subTurn][:3, :3], new_trans_matrix[:3, :3]
                )  # Rotation
                this_trans_matrix[:3, 3:4] = (
                    np.dot(
                        transfrom_for_subs[subTurn][:3, :3], new_trans_matrix[:3, 3:4]
                    )
                    + transfrom_for_subs[subTurn][:3, 3:4]
                )  # Translation
                transfrom_for_subs[line_content[-1]] = this_trans_matrix

                continue

            """Detect the declaration of a brick"""
            brick_id = line_content[-1][0:-4]
            if brick_id in template_ids:

                # processing brick color
                color = int(line_content[1])

                # processing the transformation matrix
                brick_idx = template_ids.index(brick_id)
                trans_matrix = np.identity(4, dtype=float)
                new_translate = np.zeros((3, 1))

                for i in range(3):
                    new_translate[i] = float(line_content[i + 2])

                new_rotation = np.identity(3, dtype=float)
                for i in range(9):
                    new_rotation[i // 3][i % 3] = float(line_content[i + 5])

                trans_matrix[:3, 3:4] = (
                    np.dot(transfrom_for_subs[subTurn][:3, :3], new_translate)
                    + transfrom_for_subs[subTurn][:3, 3:4]
                )
                trans_matrix[:3, :3] = np.dot(
                    transfrom_for_subs[subTurn][:3, :3], new_rotation
                )
                brickInstance = BrickInstance(
                    brick_templates[brick_idx], np.identity(4, dtype=float), color
                )

                # print(f"rotate{trans_matrix[:3,:3]}")
                brickInstance.rotate(trans_matrix[:3, :3])
                # print(f"translate is{trans_matrix[:3, 3:4]}")
                brickInstance.translate(trans_matrix[:3, 3:4])

                """Following code is for connecting points debugging"""
                """for cp in brickInstance.get_current_conn_points():
                    #print(f"Connecting point position:{cp.pos}")
                    #print(f"Connecting point orientation:{cp.orient}")

                    testbrickinstance = BrickInstance(brick_templates[template_ids.index("18654")], np.identity(4, dtype=float), color)

                    testbrickinstance.rotate(rot_matrix_from_A_to_B(brick_templates[template_ids.index("18654")].c_points[0].orient, cp.orient))
                    testbrickinstance.translate(20 * cp.pos)
                    #print(f"rotation matrix is: {testbrickinstance.trans_matrix}")
                    bricks.append(testbrickinstance)"""

                bricks.append(brickInstance)
                print(f"brick {brickInstance.template.id} processing done")
            else:
                print(f"unrecognized brick, ID: {line_content[-1]}")
    f.close()

    return bricks
