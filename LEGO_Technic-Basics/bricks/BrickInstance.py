from Abstract.Abstract_parts import Line
from bricks.BrickTemplate import BrickTemplate
import numpy as np

from bricks.ConnPoint import CPoint

class BrickInstance():

    def __init__(self, template: BrickTemplate, trans_matrix, color = 15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color

    def to_ldraw(self):
        text = f"1 {self.color} {self.trans_matrix[0][3]} {self.trans_matrix[1][3]} {self.trans_matrix[2][3]} " + \
                f"{self.trans_matrix[0][0]} {self.trans_matrix[0][1]} {self.trans_matrix[0][2]} " + \
                f"{self.trans_matrix[1][0]} {self.trans_matrix[1][1]} {self.trans_matrix[1][2]} " + \
                f"{self.trans_matrix[2][0]} {self.trans_matrix[2][1]} {self.trans_matrix[2][2]} " + \
                f"{self.template.id}.dat"
        return text

    def rotate(self, rot_mat):
        self.trans_matrix[:3,:3] = np.dot(rot_mat, self.trans_matrix[:3,:3])

    def translate(self, trans_vec):
        self.trans_matrix[:3,3:4] = self.trans_matrix[:3,3:4] + np.reshape(trans_vec,(3,1))

    def reset_transformation(self):
        self.trans_matrix = np.identity(4,dtype=float)



    def get_current_conn_points(self):
        conn_points = []

        for cp in self.template.c_points:
            #print(self.trans_matrix[:3,:3])
            #print(cp.orient)
            conn_point_orient = np.dot(self.trans_matrix[:3,:3], cp.orient)
            #print(conn_point_orient)
            conn_point_position = np.reshape(np.dot(self.trans_matrix[:3,:3], np.reshape(cp.pos,(3,1))),(1,3))
            conn_point_position = conn_point_position + np.reshape(self.trans_matrix[:3, 3:4]/20,(1,3))
            conn_points.append(CPoint(conn_point_position, conn_point_orient, cp.type))

        return conn_points

    def get_conn_type_by_pos(self, point):
        conn_points = self.get_current_conn_points()
        for conn_point in conn_points:
            if (conn_point.pos == point).all():
                return conn_point.type




    def get_current_end_conn_points(self):
        lines = []
        anchor_points = []
        for cp in self.get_current_conn_points():
            if len(lines) == 0:
                #print(f"notice the original points at:{cp.pos}",flush=True)
                new_line = Line()
                new_line.add_points(cp)
                lines.append(new_line)
                continue
            numberoflines = len(lines)
            for i in range(len(lines)):
                for j in range(len(lines[i].conn_points)):
                    distance = np.linalg.norm(lines[i].conn_points[j].pos - cp.pos)
                    if distance <= 1.01 and distance > 0.1:
                        #print(f"notice a nearby point:{cp.pos}",flush=True)
                        if np.all(lines[i].direction == 0):

                            lines[i].direction = lines[i].conn_points[j].pos - cp.pos / distance
                            #print(f"notice a nearby point with no original direction at {cp.pos}, then initialize the direction:{lines[i].direction}",flush=True)
                            lines[i].add_points(cp)
                            break
                        elif np.abs(np.abs(np.dot(lines[i].direction, np.reshape(cp.pos - lines[i].conn_points[j].pos,(3,1))))- np.linalg.norm(lines[i].direction)
                                    * np.linalg.norm(cp.pos - lines[i].conn_points[j].pos)) < 0.01:
                            #print(f"notice a nearby point parallel to the line:{cp.pos}",flush=True)
                            lines[i].add_points(cp)
                            break
                        elif distance > 0.1:
                            n_line = Line()
                            n_line.add_points(lines[i].conn_points[j])
                            flag = 0
                            for anchor_point in anchor_points:
                                if (lines[i].conn_points[j].pos == anchor_point).all():
                                    flag = 1
                            if flag == 0:
                                anchor_points.append(lines[i].conn_points[j].pos)
                            n_line.add_points(cp)
                            n_line.direction = lines[i].conn_points[j].pos - cp.pos / distance
                            #print(f"notice a new line at {cp.pos} with new direction:{n_line.direction}",flush=True)
                            lines.append(n_line)
                        else:
                            print("Wow, a wired condition happens!")

        for line1 in lines:
            for line2 in lines:
                if line1 != line2:
                    if len(list(set(line1.conn_points).intersection(set(line2.conn_points)))) > 1:
                        line1.conn_points = list(set(line1.conn_points).union(set(line2.conn_points)))
                        lines.remove(line2)


        '''for line in lines:
            print(f"this is a lien with direction:{line.direction}")
            print(" with points at:",flush=True)
            for cp in line.conn_points:
                print(cp.pos)'''
        #print("now the start points and end points are:")
        start_points = []
        end_points = []

        for line in lines:
            start_point, end_point = line.get_end_points()
            start_points.append(start_point)
            end_points.append(end_point)

        '''for start_point in start_points:
            print(start_point)
        for end_point in end_points:
            print(end_point)'''
        #print(start_points)
        #print(end_points)

        return start_points, end_points, anchor_points






