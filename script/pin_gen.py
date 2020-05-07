from itertools import combinations

edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
]

pins_map = {i: [] for i in range(8)}
for i, (p, q) in enumerate(edges):
    pins_map[p].append(i)
    pins_map[q].append(i)

merger = lambda accum, cur: accum + cur

from functools import reduce
result = [(i, combinations(incident_edges, 2)) for i, incident_edges in pins_map.items()]

for e in edges:
    print(*e)

for i, comb in result:
    for c in comb:
        print(i, *c)
