#include <cstdio>
#include <Eigen>

#ifndef READER_H
#define READER_H
bool readDataFile(
    const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins,
    Eigen::MatrixXi &anchors
);
#endif