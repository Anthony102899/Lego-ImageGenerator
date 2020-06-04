#include <Eigen>

#ifndef SHIFTER_H
#define SHIFTER_H

Eigen::MatrixXd displacementMatrix(
    Eigen::MatrixXd P, 
    Eigen::MatrixXi E, 
    Eigen::VectorXd x_shift);
Eigen::MatrixXd shift(
    Eigen::MatrixXd P, 
    Eigen::MatrixXi E, 
    Eigen::VectorXd x_shift, 
    double t);

#endif 