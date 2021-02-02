import pickle
import numpy as np

import util.geometry_util as geo_util

test_thetaphi = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 2],
]) * np.pi / 4

print(geo_util.unitsphere2cart(test_thetaphi))

with open("uniform.pickle", "rb") as fp:
    data = pickle.load(fp)
    eigenpair = [(item["objective"], np.array(item["axes"], dtype=np.double)) for item in data]

max_pair = max(eigenpair, key=lambda p: p[0])
min_pair = min(eigenpair, key=lambda p: p[0])
print(max_pair, min_pair, sep='\n')
