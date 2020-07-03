import json
from os import path

from util.debugger import MyDebugger


def reflect_y(data):
    for brick in data:
        for connPoint in brick["connPoint"]:
            connPoint["pos"][1] *= -1


def swap_xz(data):
    for brick in data:
        if brick["id"] in {"2456"}:
            for connPoint in brick["connPoint"]:
                temp = connPoint["pos"][0]
                connPoint["pos"][0] = connPoint["pos"][2]
                connPoint["pos"][2] = temp


# for temporaty fix errors in the database
if __name__ == "__main__":
    debugger = MyDebugger("test")
    database_file = path.join(
        path.dirname(path.dirname(__file__)),
        "database",
        "regular_brick_database.json",
    )
    with open(database_file) as f:
        data = json.load(f)
    reflect_y(data)

    print(json.dumps(data))
