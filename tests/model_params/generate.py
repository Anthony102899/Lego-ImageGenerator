import json

def render(points, axes):
    return {
        "beams": [
            {
                "id": "b0",
                "nodes": [
                    points[0],
                    points[1],
                ],
                "thickness": 1,
            },
            {
                "id": "b1",
                "nodes": [
                    points[1],
                    points[2],
                ],
                "thickness": 1,
            },
            {
                "id": "b2",
                "nodes": [
                    points[2],
                    points[3]
                ],
                "thickness": 1,
            },
            {
                "id": "b3",
                "nodes": [
                    points[3],
                    points[0],
                ],
                "thickness": 1,
            },
        ],
        "hinges": [
            {
                "id": f"h{i}",
                "beams": [f"b{i}", f"b{(i + 1) % 4}"],
                "pivot_point": points[(i + 1) % 4],
                "axis": axes[i],
            } for i in range(4)
        ],
    }

axes_set = [
    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ],
    [
        [1, 1, 0],
        [-1, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ]
]

for index, axes in enumerate(axes_set):
    points = [
        [0, 0, 0],
        [0, 5, 0],
        [5, 5, 0],
        [5, 0, 0],
    ]

    obj = render(points, axes)

    with open(f"square-{index}.json", "w") as fp:
        json.dump(obj, fp, indent=2)