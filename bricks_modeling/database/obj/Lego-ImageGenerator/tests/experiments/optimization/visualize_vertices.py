from matplotlib import pyplot as plt
import numpy as np

filename = "another_bridge_output.npz"
vertices = np.load(filename)["arr_0"]

for vertex_ind in (4, 5):
    vertex = vertices[:, vertex_ind, :]
    plt.plot(vertex[:, 0], vertex[:, 1], color="black")
    plt.axis("equal")
    plt.axis("off")

# plt.show()
plt.savefig("bridge.svg", transparent=True)
