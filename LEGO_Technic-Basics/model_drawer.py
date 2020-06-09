import open3d as o3d
from Abstract.Abstract_parts import Points, Edges, Anchors, Pins, Line, hash_for_edge, Axles


def draw(points: Points, edges: Edges, anchors: Anchors, pins: Pins, axles:Axles):
    points_to_draw = []
    for point in points.points:
        points_to_draw.append(point[0].tolist())
    #print(points_to_draw)
    edges_to_draw = []
    for edge in edges.edges:
        edges_to_draw.append(edge)
    #print(edges_to_draw)
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points_to_draw),
        lines = o3d.utility.Vector2iVector(edges_to_draw),
    )
    colors = []
    for edge in edges.edges:
        flag = 0
        for pin in pins.pins:
            if edges.edegs_to_index[hash_for_edge(edge)] == pin[1]:
                print(f"draw pin{pin}")
                #colors.append([1,0,0])
                flag = 1
                #break
        for axle in axles.axles:
            if edges.edegs_to_index[hash_for_edge(edge)] == axle[2]:
                print(f"draw axle{axle}")
                flag = 2
        if flag == 0:
            colors.append([0,0,0])
        elif flag == 1:
            colors.append([1,0,0])
        elif flag == 2:
            colors.append([0,1,0])


    #colors = [[1, 0, 0] for i in range(len(edges_to_draw))]


    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])

'''if __name__ == "__main__":
    print("Let's draw a cubic using o3d.geometry.LineSet.")
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])'''