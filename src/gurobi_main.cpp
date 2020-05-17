#include "gurobi_solver.h"
#include "reader.h"
#include "solver.h"
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

    solve_by_gurobi(C, b, P, E);

    return 0;
}