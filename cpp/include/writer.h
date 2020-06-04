#include <Eigen>

#include <nlohmann/json.hpp>

#include "gurobi_solver.h"

using json = nlohmann::json;

#ifndef WRITER_H
#define WRITER_H
void writeMatrixToCsv(std::string filename, Eigen::MatrixXd M);
void writeMatrices(const char *filename, std::vector<Eigen::MatrixXd> matrices);
void writeObjSolPairsJson(std::string filename, std::vector<ObjSolPair> pairs);
json parseObjSolPairsToJson(std::vector<ObjSolPair> pairs);
void writeJsonToFile(std::string filename, json j);
void writeObjSolPairs2dJson(
    std::string filename,
    std::vector<double> epsilons,
    std::vector<double> costs, 
    std::vector<std::vector<ObjSolPair>> pairs2d);
#endif