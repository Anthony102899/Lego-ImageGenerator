#include <Eigen>

#ifndef LP_SOLVER_H
#define LP_SOLVER_H

double solve_by_gurobi(
    Eigen::MatrixXd C, 
    Eigen::VectorXd b, 
    Eigen::MatrixXd V,
    Eigen::MatrixXi E,
    bool verbose = true);

#endif
