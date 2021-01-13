import numpy as np
from util import geometry_util as gu
from solvers.rigidity_solver import algo_core, visualization as vis

if __name__ == "__main__":
    P = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]) * 10 + 5

    E = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        # [2, 3],
    ])

    Q = algo_core.spring_energy_matrix(P, E)
    pairs = gu.eigen(Q, symmetric=True)
    evalues, evectors = zip(*pairs)

    print("nullity", len(P) * 3 - np.linalg.matrix_rank(Q))
    print("rank", np.linalg.matrix_rank(Q))

    basis = gu.trivial_basis(P)
    # rawbasis = gu.trivial_basis(P, orthonormal=False)
    # q, r = np.linalg.qr(rawbasis.T)
    # print("q", q, "r", r, sep="\n")
    reduced_zeroeigenvectors = [gu.subtract_orthobasis(vec, basis) for val, vec in pairs if abs(val) < 1e-9]

    
    # print(evectors[0])
    # print(reduced_eigenvectors[0])

    coeffs = gu.decompose_on_orthobasis(evectors[0], basis)

    sumvector = np.sum(reduced_zeroeigenvectors, axis=0)

    print(sumvector.reshape((-1, 3)).sum(axis=0))
    # for vec in reduced_zeroeigenvectors:
    #     print(vec)

    reduced_zerospace = np.array(reduced_zeroeigenvectors)
    # print(gu.rref(reduced_zerospace))
    # print(np.linalg.matrix_rank(reduced_zerospace))

    visvector = sumvector
    print(visvector)

    # vis.show_graph(P, E, sumvector.reshape((-1, 3)))
    vis.show_graph(P, E, visvector.reshape((-1, 3)))

    # for val, vec in pairs:
    #     if abs(val) > 1e-6:
    #         print("vec", vec, "r-vec", gu.subtract_orthobasis(vec, basis), sep="\n")



 