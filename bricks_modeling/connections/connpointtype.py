import enum
import math


class ConnPointType(enum.Enum):
    HOLE = 1
    PIN = 2
    AXLE = 3
    CROSS_HOLE = 4
    SOLID = 5
    STUD = 6
    TUBE = 7


stringToType = {
    "hole": ConnPointType.HOLE,
    "pin": ConnPointType.PIN,
    "axle": ConnPointType.AXLE,
    "cross_hole": ConnPointType.CROSS_HOLE,
    "solid": ConnPointType.SOLID,
    "stud": ConnPointType.STUD,
    "tube": ConnPointType.TUBE
}

# type to if the connection point is valid in both sides along the normal
isDoubleOriented = {
    ConnPointType.HOLE :  True,
    ConnPointType.PIN :   False,
    ConnPointType.AXLE :  True,
    ConnPointType.CROSS_HOLE :  True,
    ConnPointType.SOLID :  True,
    ConnPointType.STUD : False,
    ConnPointType.TUBE : False
}

# type to (length along the normal, and two lateral direction of the normal)
typeToBoundingBox = {
    ConnPointType.HOLE : (18.5, 19, 19),
    ConnPointType.PIN :  (20, 20, 20),
    ConnPointType.AXLE : (20, 20, 20),
    ConnPointType.CROSS_HOLE : (20, 20, 20),
    ConnPointType.SOLID : (20, 20, 20),
    ConnPointType.STUD : (3.8, 11, 11),
    ConnPointType.TUBE : (3.8, 16, 16)
}

# these properties are for visualization ONLY
# property: (debug brick, orientation in local coordinate, offset of the center, scaling in three directions)
typeToBrick = {
    ConnPointType.HOLE: ("18654.dat", [0, 1, 0], [0, 0, 0], [1, 1, 1]),
    ConnPointType.PIN:  ("4274.dat", [1, 0, 0], [10, 0, 0], [1, 1, 1]),
    ConnPointType.AXLE: ("3704.dat", [1, 0, 0], [0, 0, 0], [0.5, 1, 1]),
    ConnPointType.CROSS_HOLE: ("axle.dat", [0, 1, 0], [0, -10, 0], [1, 20, 1]),
    ConnPointType.SOLID: ("99948.dat", [0, 1, 0], [0, 0, 0], [0.2225, 0.2225, 0.2225]),
    ConnPointType.STUD:  ("stud.dat", [0, 1, 0], [0, 2, 0], [1, 1, 1]),
    ConnPointType.TUBE:  ("box5.dat", [0, 1, 0], [0, 0.5, 0], [6, 1, 6])
}

if __name__ == "__main__":
    print(isDoubleOriented[ConnPointType.TUBE])