import numpy as np

from Abstract.Abstract_parts import Points, Pins, Anchors, Edges, Axles


def find_intersection_in_bricks(bricks):
    pins_pos = []
    pins_from_brick = []
    pins_to_brick = []
    axle_axlehole_pos = []
    axles_axleholes_from_brick = []
    axles_axleholes_to_brick = []
    axle_hole_pos = []
    axles_holes_from_brick = []
    axles_holes_to_brick = []

    for brick1 in bricks:
        for brick2 in bricks:
            for conn_point_i in brick1.get_current_conn_points():
                for conn_point_j in brick2.get_current_conn_points():
                    if (
                            conn_point_i.pos == conn_point_j.pos).all() and conn_point_i.type == 1 and conn_point_j.type == 2 and brick1 != brick2:
                        print(f"pin {brick2.template.id} insert hole{brick1.template.id} at pos{conn_point_i.pos} ")
                        pins_pos.append(conn_point_i.pos)
                        pins_from_brick.append(brick2)
                        pins_to_brick.append(brick1)

                    if (
                            conn_point_i.pos == conn_point_j.pos).all() and conn_point_i.type == 4 and conn_point_j.type == 3 and brick1 != brick2:
                        print(
                            f"axle {brick2.template.id} insert axlehole{brick1.template.id} at pos{conn_point_i.pos} ")
                        axle_axlehole_pos.append(conn_point_i.pos)
                        axles_axleholes_from_brick.append(brick2)
                        axles_axleholes_to_brick.append(brick1)

                    if (
                            conn_point_i.pos == conn_point_j.pos).all() and conn_point_i.type == 1 and conn_point_j.type == 3 and brick1 != brick2:
                        print(f"axle {brick2.template.id} insert hole{brick1.template.id} at pos{conn_point_i.pos} ")
                        axle_hole_pos.append(conn_point_i.pos)
                        axles_holes_from_brick.append(brick2)
                        axles_holes_to_brick.append(brick1)

    return pins_pos, pins_from_brick, pins_to_brick, axle_axlehole_pos,axles_axleholes_from_brick,axles_axleholes_to_brick,axle_hole_pos,axles_holes_from_brick,axles_holes_to_brick


def find_edge_by_point(brick, point):
    start_points, end_points, anchor_points = brick.get_current_end_conn_points()
    for i in range(len(start_points)):
        v1 = start_points[i] - point
        v2 = point - end_points[i]
        if (np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0):
            return [start_points[i], end_points[i]]
        elif ((v1) / np.linalg.norm(v1)).all() == (v2 / np.linalg.norm(v2)).all():
            # print(start_points[i], end_points[i],point)
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
    axles = Axles()
    current_conn_points = []
    for brick in bricks:
        start_points, end_points, anchor_points = brick.get_current_end_conn_points()

        for point in start_points:
            points.add(point)
        for point in end_points:
            points.add(point)

        # print(points.points_to_index.keys())

        for i in range(len(start_points)):
            edges.add(start_points[i], end_points[i], points)

        # print(edges.edegs_to_index.keys())

        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                for j in range(len(end_points)):
                    if i != j and (start_points[i] == anchor_point).all() and (end_points[j] == anchor_point).all():
                        anchors.add([(start_points[i]), (end_points[i])], [(start_points[j]), (end_points[j])],
                                    anchor_point, points, edges)

        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                if (start_points[i] == anchor_point).all():
                    for j in range(len(start_points)):
                        v1 = start_points[j] - anchor_point
                        v2 = anchor_point - end_points[j]
                        if i != j and ((v1) / np.linalg.norm(v1)).all() == (v2 / np.linalg.norm(v2)).all():
                            anchors.add(([(start_points[i]), (end_points[i])]), [(start_points[j]), (end_points[j])],
                                        anchor_point, points, edges)

        for anchor_point in anchor_points:
            for i in range(len(start_points)):
                if (end_points[i] == anchor_point).all():
                    for j in range(len(start_points)):
                        v1 = start_points[j] - anchor_point
                        v2 = anchor_point - end_points[j]
                        if i != j and ((v1) / np.linalg.norm(v1) == v2 / np.linalg.norm(v2)).all():
                            anchors.add([(start_points[i]), (end_points[i])], [(start_points[j]), (end_points[j])],
                                        anchor_point, points, edges)

    '''points.print()
    edges.print(points)
    anchors.print()'''

    pins_pos, pins_from_brick, pins_to_brick, axle_axlehole_pos,axles_axleholes_from_brick,axles_axleholes_to_brick,axle_hole_pos,axles_holes_from_brick,axles_holes_to_brick\
        = find_intersection_in_bricks(bricks)
    # print(f"pins_pos is {pins_pos}")
    for i in range(len(pins_pos)):
        print("pin pass")
        points.add(pins_pos[i])
        pins.add(find_edge_by_point(pins_from_brick[i], pins_pos[i]), find_edge_by_point(pins_to_brick[i], pins_pos[i]),
                 pins_pos[i], points, edges)

    for i in range(len(axle_axlehole_pos)):
        print("axle pass")
        points.add(axle_axlehole_pos[i])
        axles.add(find_edge_by_point(axles_axleholes_from_brick[i], axle_axlehole_pos[i]), find_edge_by_point(axles_axleholes_to_brick[i], axle_axlehole_pos[i]),
                 axle_axlehole_pos[i], points, edges)


    #pins.print()

    return points, edges, anchors, pins, axles