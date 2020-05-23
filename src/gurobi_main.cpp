#include "gurobi_solver.h"
#include "reader.h"
#include "solver.h"
#include "coordinator.h"

#include "gurobi_c++.h" 
#include <iostream>


using namespace Eigen;

int main(int argc, char *argv[]) {
    MatrixXd P;
    MatrixXi E;
    MatrixXi pins;
    MatrixXi anchors;

    std::string data_file;
    if (argc < 2) {
        data_file = "data/square.txt";
    } else {
        data_file = argv[1];
    }

    read_data_file(data_file.c_str(), P, E, pins, anchors);

    MatrixXd C = build_constraints_matrix(P, E, pins, anchors);
    VectorXd b = VectorXd::Zero(C.rows());

    fix_one_edge(0, Eigen::VectorXd::Zero(6), C, b);

    bool verbose = false;
    GurobiSolver solver(C, b, P, E, verbose);

    std::vector<ObjSolPair> pairs = solveUsingL1Norm(solver, 0.01, 0.0001);

    for (unsigned i = 0; i < pairs.size(); i++) {
        std::cout << pairs.at(i).first << std::endl;
    }

    return 0;
}