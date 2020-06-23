import numpy as np

from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import ConnPointType

from typing import Set


class BrickTemplate:
    def __init__(self, c_points, ldraw_id: str):
        self.c_points = c_points
        self.id = ldraw_id

    def __eq__(self, other):
        if isinstance(other, BrickTemplate):
            return self.id == other.id
        return False
    
    def deg1_cpoints_indices(self) -> Set[int]:
        """
        return a set of the indices of the c_points that have exactly one c_point having 1 lego distance to it

        Note: '1 lego distance' is hard-coded as 20 here temporarily, which stands for a beam.
        """
        from itertools import combinations
        from collections import defaultdict
        deg_count = defaultdict(lambda: 0)
        lego_dist = 20
        tol = 1e-6

        point_positions = [pt.pos for pt in self.c_points]
        # iterate over all pairs of conn_points of the instance
        for (i, p), (j, q) in combinations(enumerate(point_positions), 2):
            if -tol < np.linalg.norm(p - q) - lego_dist < tol: # if distance within tolerance
                deg_count[i] += 1
                deg_count[j] += 1

        deg1set = {ind for ind, count in deg_count.items() if count == 1}
        return deg1set


if __name__ == "__main__":
    cpoints = [
        CPoint(np.array([0, 0, -1]), np.array([0, 1, 0]), ConnPointType.AXLE),
        CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), ConnPointType.AXLE),
        CPoint(np.array([0, 0, 1]), np.array([0, 1, 0]), ConnPointType.AXLE),
    ]
    brick = BrickTemplate(cpoints, ldraw_id="32523.dat")
    input("")
