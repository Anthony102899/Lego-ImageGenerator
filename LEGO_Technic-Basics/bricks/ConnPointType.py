import enum

class ConnPointType(enum.Enum):
   HOLE = 1
   PIN = 2
   AXLE = 3
   AXLE_HOLE = 4
   SOLID = 5

ConnTypeToJoints = {
    ConnPointType.HOLE:      {"connhole", "connhol2", "connhol3", "beamhole", "beamhol2"},
    ConnPointType.PIN:       {"connect", "connect8", "connect2", "connect3", "connect5", "connect7", "confric", "confric2", "confric5", "confric4", "confric6", "confric9", "confric8"},
    ConnPointType.AXLE:      {},
    ConnPointType.AXLE_HOLE: {},
    ConnPointType.SOLID:     {}
}

if __name__ == "__main__":
    for c in ConnPointType:
        print(c)