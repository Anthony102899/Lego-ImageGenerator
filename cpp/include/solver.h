#include <Eigen>

#ifndef SOLVER_H
#define SOLVER_H
Eigen::MatrixXd buildConstraintMatrix(
    Eigen::MatrixXd P,
    Eigen::MatrixXi E,
    Eigen::MatrixXi pins,
    Eigen::MatrixXi anchors
);

void fixOneEdge(
    int index, 
    Eigen::VectorXd vw, 
    Eigen::MatrixXd &C, 
    Eigen::VectorXd &b);

bool solve(
    Eigen::MatrixXd P,
    Eigen::MatrixXi E,
    Eigen::MatrixXi pins,
    Eigen::MatrixXi anchors,
    int &dof,
    Eigen::MatrixXd &constraints,
    std::vector<std::tuple<int, Eigen::VectorXd, double>> &unstable_indices);
#endif
