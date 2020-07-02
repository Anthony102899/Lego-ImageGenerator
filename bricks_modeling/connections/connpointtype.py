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
    "tube": ConnPointType.TUBE,
}

# property: (debug brick, orientation, offset of the center, scale, scaling direction)
typeToBrick = {
    ConnPointType.HOLE: ("18654.dat", [0, 1, 0], [0, 0, 0], 1, [0, 1, 0]),
    ConnPointType.PIN: ("4274.dat", [1, 0, 0], [10, 0, 0], 1, [0, 1, 0]),
    ConnPointType.AXLE: ("3704.dat", [1, 0, 0], [0, 0, 0], 0.5, [1, 0, 0]),
    ConnPointType.CROSS_HOLE: ("axle.dat", [0, 1, 0], [0, -10, 0], 20, [0, 1, 0]),
    ConnPointType.SOLID: ("99948.dat", [0, 1, 0], [0, 0, 0], 0.2225, [1, 1, 1]),
    ConnPointType.STUD: ("stud.dat", [0, 1, 0], [0, 2, 0], 1, [0, 1, 0]),
    ConnPointType.TUBE: ("box5.dat", [0, 1, 0], [0, 0.5, 0], 1, [0, 1, 0]),
}

if __name__ == "__main__":
    for c in ConnPointType:
        print(c)
