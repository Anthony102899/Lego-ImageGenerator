#include "gurobi_solver.h"
#include "reader.h"
#include "solver.h"
#include "coordinator.h"
#include "writer.h"

#include "gurobi_c++.h" 
#include <iostream>

using namespace std;

// declaration
string formatOutputName(string dataName, string outDir);

int main(int argc, char *argv[]) {
    Eigen::MatrixXd P;
    Eigen::MatrixXi E;
    Eigen::MatrixXi pins;
    Eigen::MatrixXi anchors;

    string dataFile;
    string outDir;
    if (argc < 2) {
        dataFile = "data/square.txt";
    } else {
        dataFile = argv[1];
    }

    if (argc < 3) { 
        outDir = "./"; 
    } else {
        outDir = argv[2];
        outDir += (outDir[-1] == '/' ? "" : "/");
    }
    
    cerr << "Running ";
    for (int i = 0; i < 3; i++) 
        cerr << argv[i] << " ";
    cerr << endl;

    read_data_file(dataFile.c_str(), P, E, pins, anchors);

    Eigen::MatrixXd C = build_constraints_matrix(P, E, pins, anchors);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(C.rows());

    fix_one_edge(0, Eigen::VectorXd::Zero(6), C, b);

    bool verbose = false;
    GurobiSolver solver(C, b, P, E, verbose);

    double epsilon = 0.01;
    double lb = 1e-6;
    double ub = 1e-2;
    cerr << "lb ~ " << lb << " ub ~ " << ub << endl;

    Eigen::VectorXd costRange = Eigen::VectorXd::LinSpaced(100, lb, ub);
    vector<double> costs = vector<double>(costRange.data(), costRange.data() + 100);
    vector<double> epsilons(100, 0.01);
    vector<vector<ObjSolPair>> pairs2d;

    for (int i = 0; i < costRange.size(); i++) {
        double cost = costs[i];
        pairs2d.push_back(solveUsingL1Norm(solver, 0.01, cost));
    }
    string name = formatOutputName(dataFile, outDir);
    writeObjSolPairs2dJson(name, epsilons, costs, pairs2d);

    return 0;
}

string formatOutputName(string dataName, string outDir) {

    string s = dataName;
    size_t pos = 0;
    string deli = "/";
    while ((pos = s.find(deli)) != string::npos) {
        s.erase(0, pos + deli.length());
    }

    ostringstream stringStream;
    stringStream << outDir << s
                 << ".json";
    return stringStream.str();
}
