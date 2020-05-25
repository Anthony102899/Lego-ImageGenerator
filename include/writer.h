#include <Eigen>

#include "gurobi_solver.h"

#ifndef WRITER_H
#define WRITER_H
void writeMatrices(const char *filename, std::vector<Eigen::MatrixXd> matrices);
void writeObjSolPairsJson(std::string filename, std::vector<ObjSolPair> pairs);
void writeObjSolPairs2dJson(
    std::string filename,
    std::vector<double> epsilons,
    std::vector<double> costs, 
    std::vector<std::vector<ObjSolPair>> pairs2d);
#endif