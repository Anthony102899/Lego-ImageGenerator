from matplotlib import pyplot as plt
import numpy as np
from model_overview import define, define_from_file

definition = define_from_file()

model = definition["model"]


# for index, beam in enumerate(model.beams):
#     plt.clf()
#     plt.axis('off')
#     points = beam.points
#     edges = beam.edges
#     for i, j in edges:
#         vertices = np.take(points, (i, j), axis=0)
#         plt.plot(vertices[:, 0], vertices[:, 1], color=(0, 0, 0))
#
#     ax = plt.gca()
#     ax.set_aspect('equal')
#     plt.savefig(f"part-{index}.png", dpi=500, transparent=True)
#
# np.savez(f"data/overview_{i}.npz",
#          eigenvalue=np.array(e),
#          points=points,
#          edges=edges,
#          eigenvector=eigenvector,
#          force=force,
#          stiffness=M)

points = model.point_matrix()
edges = model.edge_matrix()
for index in (0, 1, 2):
    plt.clf()
    plt.axis('off')

    # for i, j in edges:
    #     vertices = np.take(points, (i, j), axis=0)
    #     plt.plot(vertices[:, 0], vertices[:, 1], color=(0, 0, 0))

    data = np.load(f"data/overview_{index}.npz")
    eigenvalue, eigenvector = data["eigenvalue"], data["eigenvector"].reshape(-1, 3)

    print(index, eigenvalue)

    xs, ys = points[:, 0], points[:, 1]

    eigenvector *= 30
    dxs, dys = eigenvector[:, 0], eigenvector[:, 1]

    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        if np.linalg.norm([dx, dy]) < 1e-2:
            continue
        plt.gca().arrow(
            x, y, dx, dy,
            # length_includes_head=True,
            color=(1, 0, 0),
            width=0.5,
        )

    # plt.show()
    plt.savefig(f"eigenvector-{index}.png", dpi=500, transparent=True)
