import os


def get_vertices_2D(id):
    obj_file_path = "./database/obj/" + id + ".obj"
    vertices = []
    min_y = 0.0
    stud_vertices = []
    for line in open(obj_file_path, "r"):
        if line.startswith('v'):
            values = line.split(" ")
            if float(values[2]) < min_y:
                min_y = float(values[2])
            if round(float(values[2]), 1) == 0.0:
                v = [float(x) for x in values[1: 4]]
                vertices.append(v)
    if min_y < 0:
        for line in open(obj_file_path, "r"):  # detect vertices of studs, which will not be count as 2d vertices
            if line.startswith('v'):
                values = line.split(" ")
                if float(values[2]) == min_y:
                    v = [float(x) for x in values[1: 4]]
                    v[1] = 0
                    stud_vertices.append(v)
    result = []
    for v in vertices:
        if result.count(v) == 0 and stud_vertices.count(v) == 0:
            result.append(v)
    return result

if __name__ == "__main__" :
    print(get_vertices_2D("43722"))
