import enum

class ConnPointType(enum.Enum):
   HOLE = 1
   PIN = 2
   AXLE = 3
   CROSS_HOLE = 4
   SOLID = 5

stringToType = {"hole":ConnPointType.HOLE,
                "pin":ConnPointType.PIN,
                "axle":ConnPointType.AXLE,
                "cross_hole":ConnPointType.CROSS_HOLE,
                "solid":ConnPointType.SOLID}

if __name__ == "__main__":
    for c in ConnPointType:
        print(c)