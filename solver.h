#include <Eigen>

#ifndef SOLVER_H
#define SOLVER_H
bool solve(
    Eigen::MatrixXd P,
    Eigen::MatrixXi E,
    Eigen::MatrixXi pins,
    int &dof,
    std::vector<std::pair<int, Eigen::VectorXd>> &unstable_indices);
#endif
