#include "writer.h"
#include "reader.h"
#include "solver.h"

#include <Eigen>

using namespace std;

int main(int argc, char *argv[]) {
    assert(argc == 3);

    string object = argv[1];
    string mat    = argv[2];

    Eigen::MatrixXd P;
    Eigen::MatrixXi E;
    Eigen::MatrixXi pins;
    Eigen::MatrixXi anchors;

    readDataFile(object.c_str(), P, E, pins, anchors);

    Eigen::MatrixXd C = build_constraints_matrix(P, E, pins, anchors);

    writeMatrixToCsv(mat, C);
}