#include <cstdio>
#include <Eigen>

#ifndef READER_H
#define READER_H
bool read_data_file(
    const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins,
    Eigen::MatrixXi &anchors
);

#endif