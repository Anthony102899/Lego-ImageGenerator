#include <cstdio>
#include <Eigen>

#ifndef READER_H
#define READER_H
bool read_data_file(
    const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins
);
void triangle_data(Eigen::MatrixXd &P, Eigen::MatrixXi &E, Eigen::MatrixXi &pins);
void square_data(Eigen::MatrixXd &P, Eigen::MatrixXi &E, Eigen::MatrixXi &pins);
void two_triangles_data(Eigen::MatrixXd &P, Eigen::MatrixXi &E, Eigen::MatrixXi &pins);
void trapezoid_data(Eigen::MatrixXd &P, Eigen::MatrixXi &E, Eigen::MatrixXi &pins);
#endif