#include <Eigen>

#ifndef SOLVER_H
#define SOLVER_H
Eigen::MatrixXd build_constraints_matrix(
    Eigen::MatrixXd P,
    Eigen::MatrixXi E,
    Eigen::MatrixXi pins,
    Eigen::MatrixXi anchors
);

bool solve(
    Eigen::MatrixXd P,
    Eigen::MatrixXi E,
    Eigen::MatrixXi pins,
    Eigen::MatrixXi anchors,
    int &dof,
    Eigen::MatrixXd &constraints,
    std::vector<std::tuple<int, Eigen::VectorXd, double>> &unstable_indices);
#endif
