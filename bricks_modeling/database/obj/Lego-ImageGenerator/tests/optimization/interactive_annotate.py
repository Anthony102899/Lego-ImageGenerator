import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util.geometry_util as geo_util


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


plt.clf()
# plt.setp(plt.gca(), autoscale_on=False)
img = mpimg.imread("excavator.png")
print(img.shape)
plt.imshow(img)
tellme('You will annotate points and edges, click to begin')

plt.waitforbuttonpress()

total_pts = []
while True:
    tellme('Point selection: select 1 point')
    [[x, y]] = np.asarray(plt.ginput(1, timeout=-1))

    ph = plt.scatter([x], [y], color=(0, 0, 0))

    tellme(f'{len(total_pts)} pts, key click for yes, mouse click for no')

    total_pts.append((x, y))
    if plt.waitforbuttonpress():
        break

total_pts = np.asarray(total_pts, dtype=np.double)

edge_set = {}
while True:
    tellme('Edge selection: select 2 points')
    pts = np.asarray(plt.ginput(2, timeout=-1))

    def closest_index(pt):
        vectors = total_pts - pt
        distances = np.linalg.norm(vectors, axis=1)
        smallest_index = np.argmin(distances)
        return smallest_index

    edge = tuple(sorted(map(int, map(closest_index, pts))))

    if edge[0] == edge[1]:
        continue

    if edge in edge_set:
        ph = edge_set[edge]
        for p in ph:
            p.remove()
        del edge_set[edge]

    else:
        selected_pts = np.take(total_pts, edge, axis=0)
        ph = plt.plot(selected_pts[:, 0], selected_pts[:, 1], color=(0, 0, 0))
        edge_set[edge] = ph

    tellme(f'{len(edge_set)} edges, key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break

plt.close()

name = input("Name for the part you annotated: ")
obj = {
    "points": total_pts.tolist(),
    "edges": list(edge_set.keys()),
}

print(obj)

with open(f"points-edges-{name}.json", "w") as fp:
    json.dump(obj, fp)
