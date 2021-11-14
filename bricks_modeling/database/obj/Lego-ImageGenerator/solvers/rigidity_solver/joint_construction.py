from bricks_modeling.connections.conn_type import ConnType
import util.geometry_util as geo_util


def construct_joints(abstract_edges, bi, bj, contact_orient, edge, edges, p, points, points_on_brick):
    # share points parallel to the contact_orient
    if edge["type"] == ConnType.HOLE_PIN.name:
        for i in range(7):
            if i in {0, 1, 2}:
                points.append(p[i])
                points_on_brick[bi].append(len(points) - 1)
                points_on_brick[bj].append(len(points) - 1)
            else:
                points.append(p[i])
                points_on_brick[bi].append(len(points) - 1)
                points.append(p[i])
                points_on_brick[bj].append(len(points) - 1)
    # share all points
    if edge["type"] == ConnType.CROSS_AXLE.name:
        for i in range(7):
            points.append(p[i])
            points_on_brick[bi].append(len(points) - 1)
            points_on_brick[bj].append(len(points) - 1)
    # share points parallel to the contact_orient
    if edge["type"] == ConnType.STUD_TUBE.name:
        for i in range(7):
            if i in {0, 1, 2}:
                points.append(p[i])
                points_on_brick[bi].append(len(points) - 1)
                points_on_brick[bj].append(len(points) - 1)
            else:
                points.append(p[i])
                points_on_brick[bi].append(len(points) - 1)
                points.append(p[i])
                points_on_brick[bj].append(len(points) - 1)
    if edge["type"] == ConnType.HOLE_AXLE.name:
        # they do not share any points
        points_base_index = len(points)
        for i in range(7):
            points.append(p[i])
            points_on_brick[bi].append(len(points) - 1)
            points.append(p[i])
            points_on_brick[bj].append(len(points) - 1)

        p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(contact_orient)
        p_vec3 = (p_vec1 + p_vec2) / 2
        abstract_edges.append([points_base_index, points_base_index + 1, p_vec1[0], p_vec1[1], p_vec1[2]])
        abstract_edges.append([points_base_index + 2, points_base_index + 3, p_vec2[0], p_vec2[1], p_vec2[2]])
        abstract_edges.append([points_base_index + 4, points_base_index + 5, p_vec3[0], p_vec3[1], p_vec3[2]])
        abstract_edges.append([points_base_index + 6, points_base_index + 7, p_vec1[0], p_vec1[1], p_vec1[2]])
        abstract_edges.append([points_base_index + 8, points_base_index + 9, -p_vec1[0], -p_vec1[1], -p_vec1[2]])
        abstract_edges.append([points_base_index + 10, points_base_index + 11, p_vec2[0], p_vec2[1], p_vec2[2]])
        abstract_edges.append([points_base_index + 12, points_base_index + 13, -p_vec2[0], -p_vec2[1], -p_vec2[2]])
    # This is a conntype not in our database
    if edge["type"] == ConnType.PRISMATIC.name:
        points_base_index = len(points)
        for i in range(7):
            points.append(p[i])
            points_on_brick[bi].append(len(points) - 1)
            points.append(p[i])
            points_on_brick[bj].append(len(points) - 1)
            edges.append([len(points) - 2, len(points) - 1])

        p_vec1, p_vec2 = geo_util.get_perpendicular_vecs(contact_orient)
        p_vec3 = (p_vec1 + p_vec2) / 2
        abstract_edges.append([points_base_index, points_base_index + 1, p_vec1[0], p_vec1[1], p_vec1[2]])
        abstract_edges.append([points_base_index + 2, points_base_index + 3, p_vec2[0], p_vec2[1], p_vec2[2]])
        abstract_edges.append([points_base_index + 4, points_base_index + 5, p_vec3[0], p_vec3[1], p_vec3[2]])
        abstract_edges.append([points_base_index + 6, points_base_index + 7, p_vec2[0], p_vec2[1], p_vec2[2]])
        abstract_edges.append([points_base_index + 8, points_base_index + 9, -p_vec2[0], -p_vec2[1], -p_vec2[2]])
        abstract_edges.append([points_base_index + 10, points_base_index + 11, p_vec1[0], p_vec1[1], p_vec1[2]])
        abstract_edges.append([points_base_index + 12, points_base_index + 13, -p_vec1[0], -p_vec1[1], -p_vec1[2]])
