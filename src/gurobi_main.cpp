#include "gurobi_solver.h"
#include "reader.h"
#include "solver.h"
#include "coordinator.h"
#include "writer.h"

#include "gurobi_c++.h" 
#include <iostream>
#include <string>

using namespace std;

// declaration
string formatOutputName(string dataName, string outDir, double eps, double cost);

int main(int argc, char *argv[]) {
    Eigen::MatrixXd P;
    Eigen::MatrixXi E;
    Eigen::MatrixXi pins;
    Eigen::MatrixXi anchors;

    string dataFile;
    string outFile;
    double eps;
    double cost;
    bool   verbose;
    if (argc < 5) {
        cerr << "Usage: gurobi_solver datafile outfile epsilon cost" << endl;
        exit(1);
    }
    dataFile = argv[1];
    outFile  = argv[2];
    eps      = std::stod(argv[3]);
    cost     = std::stod(argv[4]);
    verbose  = argc > 5 && std::stoi(argv[5]) != 0;

    if (verbose) {
        cerr << "gurobi solver running";
        for (int i = 0; i < argc; i++) 
            cerr << argv[i] << " ";
        cerr << endl;
    }

    read_data_file(dataFile.c_str(), P, E, pins, anchors);

    Eigen::MatrixXd C = build_constraints_matrix(P, E, pins, anchors);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(C.rows());

    fix_one_edge(0, Eigen::VectorXd::Zero(6), C, b);

    GurobiSolver solver(C, b, P, E, verbose);

    auto pairs = solveUsingL1Norm(solver, eps, cost);
    // string name = formatOutputName(dataFile, outDir, eps, cost);

    json result = parseObjSolPairsToJson(pairs);
    json setting;
    setting["epsilon"] = eps;
    setting["cost"] = cost;
    setting["file"] = dataFile;
    json out;
    out["setting"] = setting;
    out["result"] = result;
    writeJsonToFile(outFile, out);

    return 0;
}

string formatOutputName(string dataName, string outDir, double eps, double cost) {

    string s = dataName;
    size_t pos = 0;
    string deli = "/";
    while ((pos = s.find(deli)) != string::npos) {
        s.erase(0, pos + deli.length());
    }

    outDir += (outDir.back() != '/') ? "/" : "";

    ostringstream stringStream;
    stringStream << outDir << s
                 << "_eps_" << eps
                 << "_cost_" << cost
                 << ".json";
    return stringStream.str();
}
