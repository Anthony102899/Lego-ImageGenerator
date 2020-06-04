#include "shifter.h"

#include <iostream>
#define o(x) {std::cout << (x) << std::endl;}
#define oo(x, y) {std::cout << (x) << " " << (y) << std::endl;};

using namespace Eigen;

MatrixXd displacement_matrix(MatrixXd P, MatrixXi E, VectorXd x_shift) {
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

        for (int m = 0; m < delta_a1.size(); m++) {
            delta_a1(m) = (delta_a1(m) < 1e-11 && delta_a1(m) > -1e-11) ? 0 : delta_a1(m);
            delta_a2(m) = (delta_a2(m) < 1e-11 && delta_a2(m) > -1e-11) ? 0 : delta_a2(m);
        }
        // oo("shift edge i", i);
        // oo("p1_ind", p1_ind);
        // oo("p2_ind", p2_ind);
        // oo("p1", p1.transpose());
        // oo("p2", p2.transpose());
        // oo("v", edge_v.transpose());
        // oo("w", edge_w.transpose());
        // oo("a1", a1.transpose());
        // oo("a2", a2.transpose());
        // oo("delta_a1", delta_a1.transpose());
        // oo("delta_a2", delta_a2.transpose());
        // o("----------------")
        D.row(p1_ind) += delta_a1;
        D.row(p2_ind) += delta_a2;

        shifted[p1_ind] += 1;
        shifted[p2_ind] += 1;
    }

    for (int i = 0; i < D.rows(); i++) {
        if (shifted.at(i)) {
            D.row(i) /= shifted.at(i);
        }
    }

    return D;
}

MatrixXd shift(MatrixXd P, MatrixXi E, VectorXd x_shift, double t) {
    int items = P.rows();
    std::vector<int> shifted(items, 0);
    MatrixXd D = displacement_matrix(P, E, x_shift);
    D = D * t;
    MatrixXd P_new = P + D;
    return P_new;
}