#include <Eigen>

#ifndef WRITER_H
#define WRITER_H
void write_matrices(const char *filename, std::vector<Eigen::MatrixXd> matrices);
#endif