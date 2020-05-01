#include "shifter.h"

#include <iostream>
#define o(x) {std::cout << (x) << std::endl;}
#define oo(x, y) {std::cout << (x) << " " << (y) << std::endl;};

using namespace Eigen;

MatrixXd shift(MatrixXd P, MatrixXi E, VectorXd x_shift, double t) {
    int items = P.rows();
    std::vector<int> shifted(items, 0);
    MatrixXd D = MatrixXd::Zero(P.rows(), P.cols());

    for (int i = 0; i < E.rows(); i++) {
        int p1_ind = E(i, 0);
        int p2_ind = E(i, 1);
        Vector3d p1 = P.row(p1_ind);
        Vector3d p2 = P.row(p2_ind);
        Vector3d mid = (p1 + p2) / 2.0;

        Vector3d a1 = p1 - mid;
        Vector3d a2 = p2 - mid;

        Vector3d edge_v = x_shift.segment<3>(i * 6);
        Vector3d edge_w = x_shift.segment<3>(i * 6 + 3);

        Vector3d delta_a1 = edge_v + edge_w.cross(a1);
        Vector3d delta_a2 = edge_v + edge_w.cross(a2);

        D.row(p1_ind) += delta_a1;
        D.row(p2_ind) += delta_a2;
        shifted[p1_ind] += 1;
        shifted[p2_ind] += 1;
    }

    for (int i = 0; i < D.rows(); i++) {
        D.row(i) /= shifted.at(i);
    }

    D = D * t;
    MatrixXd P_new = P + D;
    // o(P);
    return P_new;
}