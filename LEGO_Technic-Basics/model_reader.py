import numpy as np
import json
from bricks.BrickTemplate import BrickTemplate
from bricks.ConnPoint import CPoint
from bricks.BrickInstance import BrickInstance
from Abstract.Abstract_parts import Points, Edges, Anchors, Pins, Line
from util.debugger import MyDebugger
from bricks.brick_factory import get_all_brick_templates
import model_writer
import math
import model_drawer

def rot_matrix_from_A_to_B(A, B):
    #print(A)
    #print(B)
    cross = np.cross(A, B)
    #print(f"cross is {cross}")
    dot = np.dot(A, B)
    #print(f"dot is {dot}")
    #angle = math.atan2(np.linalg.norm(cross), dot)
    angle = math.acos(dot)
    #print(f"angle is {angle}")
    rotation_axes = cross / np.linalg.norm(cross)
    #print(f"cross is {cross}")
    M = np.array([[0, -rotation_axes[2], rotation_axes[1]],
                  [rotation_axes[2], 0, -rotation_axes[0]],
                  [-rotation_axes[1], rotation_axes[0], 0]])
    if np.linalg.norm(cross) == 0:
        return np.identity(3, dtype=float)
    return np.identity(3, dtype=float) + math.sin(angle) * M + (1 - math.cos(angle)) * np.dot(M, M)


def read_bricks_from_file(file_path):
    f = open(file_path, "r")

    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    transfrom_for_subs = {}
    transfrom_for_subs["origin"] = np.identity(4, dtype=float)
    subTurn = "origin"

    for line in f.readlines():
        line_content = line.rstrip().split(" ")

        '''Detect The following lines are for another subparts'''
        if line_content[0] == "0" and "Sub" in line_content[1]:
            subTurn = line_content[1] + ".ldr"
            print(f"Notice This is a subgraph for {line_content[1]}:")
            continue

        '''Detect new delcared subparts'''
        if line_content[0] == "1":
            if "Sub" in line_content[-1]:
                print(f"Notice a declared subgraph for {line_content[-1]}")

                new_trans_matrix = np.identity(4, dtype=float)
                for i in range(3):
                    new_trans_matrix[i][3] = float(line_content[i + 2])
                for i in range(9):
                    new_trans_matrix[i // 3][i % 3] = float(line_content[i + 5])

                this_trans_matrix = np.identity(4, dtype=float)
                this_trans_matrix[:3,:3] = np.dot(transfrom_for_subs[subTurn][:3,:3], new_trans_matrix[:3,:3])#Rotation
                this_trans_matrix[:3, 3:4] = np.dot(transfrom_for_subs[subTurn][:3,:3], new_trans_matrix[:3, 3:4]) + transfrom_for_subs[subTurn][:3, 3:4]#Translation
                transfrom_for_subs[line_content[-1]] = this_trans_matrix

                continue

            '''Detect the declaration of a brick'''
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

                trans_matrix[:3, 3:4] = np.dot(transfrom_for_subs[subTurn][:3,:3], new_translate) + transfrom_for_subs[subTurn][:3, 3:4]
                trans_matrix[:3,:3] = np.dot(transfrom_for_subs[subTurn][:3,:3], new_rotation)
                brickInstance = BrickInstance(brick_templates[brick_idx], np.identity(4, dtype=float), color)

                #print(f"rotate{trans_matrix[:3,:3]}")
                brickInstance.rotate(trans_matrix[:3,:3])
                #print(f"translate is{trans_matrix[:3, 3:4]}")
                brickInstance.translate(trans_matrix[:3, 3:4])


                '''Following code is for connecting points debugging'''
                '''for cp in brickInstance.get_current_conn_points():
                    #print(f"Connecting point position:{cp.pos}")
                    #print(f"Connecting point orientation:{cp.orient}")

                    testbrickinstance = BrickInstance(brick_templates[template_ids.index("18654")], np.identity(4, dtype=float), color)

                    testbrickinstance.rotate(rot_matrix_from_A_to_B(brick_templates[template_ids.index("18654")].c_points[0].orient, cp.orient))
                    testbrickinstance.translate(20 * cp.pos)
                    #print(f"rotation matrix is: {testbrickinstance.trans_matrix}")
                    bricks.append(testbrickinstance)'''

                bricks.append(brickInstance)
                print(f"brick {brickInstance.template.id} processing done")
            else:
                print(f"unrecognized brick, ID: {line_content[-1]}")
    f.close()

    return bricks

def find_pins_in_bricks(bricks):
    pins_pos = []
    from_brick = []
    to_brick = []
    for brick1 in bricks:
        for brick2 in bricks:
            for conn_point_i in brick1.get_current_conn_points():
                for conn_point_j in brick2.get_current_conn_points():
                    if (conn_point_i.pos == conn_point_j.pos).all() and conn_point_i.type == 1 and conn_point_j.type == 2 and brick1 != brick2:
                        print(f"{brick2.template.id} insert {brick1.template.id} at pos{conn_point_i.pos} ")
                        pins_pos.append(conn_point_i.pos)
                        from_brick.append(brick2)
                        to_brick.append(brick1)

    return pins_pos, from_brick, to_brick


def find_edge_by_point(brick, point):
    start_points, end_points, anchor_points = brick.get_current_end_conn_points()
    for i in range(len(start_points)):
        v1 = start_points[i] - point
        v2 = point - end_points[i]
        if(np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0):
            return [start_points[i], end_points[i]]
        elif ((v1)/np.linalg.norm(v1)).all() == (v2/np.linalg.norm(v2)).all():
            #print(start_points[i], end_points[i],point)
            return [start_points[i], end_points[i]]

def find_brick_by_conn_points(bricks, point):
    finded_bricks = []
    for brick in bricks:
        for con_point in brick.get_current_conn_points():
            if (con_point.pos == point).all():
                finded_bricks.append(brick)
    if len(bricks) > 1:
        print("seems collision")
    else:
        return finded_bricks[0]

def find_conn_type_by_conn_points(bricks, point):
    for brick in bricks:
        for con_point in brick.get_current_conn_points():
            if (con_point.pos == point).all():
                return con_point.type


def read_features_from_bricks(bricks):
    points = Points()
    edges = Edges()
    pins = Pins()
    anchors = Anchors()
    current_conn_points = []
    for brick in bricks:
        start_points, end_points, anchor_points = brick.get_current_end_conn_points()

        for point in start_points:
            points.add(point)
        for point in end_points:
            points.add(point)

        #print(points.points_to_index.keys())

        for i in range(len(start_points)):

            edges.add(start_points[i],end_points[i], points)

        #print(edges.edegs_to_index.keys())




        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                for j in range(len(end_points)):
                    if i != j and (start_points[i] == anchor_point).all() and (end_points[j] == anchor_point).all():

                        anchors.add([(start_points[i]),(end_points[i])], [(start_points[j]),(end_points[j])], anchor_point,points,edges)

        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                if (start_points[i] == anchor_point).all():
                    for j in range(len(start_points)):
                        v1 = start_points[j] - anchor_point
                        v2 = anchor_point - end_points[j]
                        if i != j and ((v1)/np.linalg.norm(v1)).all() == (v2/np.linalg.norm(v2)).all():
                            anchors.add(([(start_points[i]), (end_points[i])]), [(start_points[j]), (end_points[j])], anchor_point, points, edges)

        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                if (end_points[i] == anchor_point).all():
                    for j in range(len(start_points)):
                        v1 = start_points[j] - anchor_point
                        v2 = anchor_point - end_points[j]
                        if i != j and ((v1)/np.linalg.norm(v1) == v2/np.linalg.norm(v2)).all():
                            anchors.add([(start_points[i]), (end_points[i])], [(start_points[j]), (end_points[j])], anchor_point, points, edges)


    points.print()
    edges.print(points)
    anchors.print()

    pins_pos, from_bricks, to_bricks = find_pins_in_bricks(bricks)
    print(f"pins_pos is {pins_pos}")
    for i in range(len(pins_pos)):
        print("pin pass")
        flag = 0
        for point in points.points:
            if (point == pins_pos[i]).all():
                flag == 1
                break
        if flag == 0:
            points.add(pins_pos[i])
        pins.add(find_edge_by_point(from_bricks[i], pins_pos[i]), find_edge_by_point(to_bricks[i], pins_pos[i]), pins_pos[i],points, edges)

    pins.print()

    return points, edges, anchors, pins




if __name__ == "__main__":
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file("./data/cube11.ldr")
    #model_writer.write_bricks_to_file(bricks, debugger.file_path("model.ldr"))
    #lines = bricks[0].get_current_end_conn_points()
    points, edges, anchors, pins = read_features_from_bricks(bricks)
    #model_writer.write_features_to_input(points, edges, anchors, pins,debugger.file_path("model.ldr"))
    model_drawer.draw(points, edges, anchors, pins)
