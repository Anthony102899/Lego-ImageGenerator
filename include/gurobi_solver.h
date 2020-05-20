#include <Eigen>
#include <memory>
#include <functional>

#include "gurobi_c++.h"

#ifndef LP_SOLVER_H
#define LP_SOLVER_H

using VertexLinObjFunctor = std::function<GRBLinExpr(GRBLinExpr, GRBLinExpr, GRBLinExpr)>;
using VertexQuadObjFunctor = std::function<GRBQuadExpr(GRBLinExpr, GRBLinExpr, GRBLinExpr)>;

struct GurobiSolver {
    Eigen::MatrixXd C;
    Eigen::VectorXd b;
    Eigen::MatrixXd V;
    Eigen::MatrixXi E;
    bool verbose;

    std::shared_ptr<GRBModel> model;
    std::vector<GRBVar> vars;

    GurobiSolver(
        Eigen::MatrixXd C,
        Eigen::VectorXd b, 
        Eigen::MatrixXd V,
        Eigen::MatrixXi E,
        bool verbose
    ) : C(C), b(b), V(V), E(E), verbose(verbose) {
        buildModel(model, vars);
    }

    bool buildModel(std::shared_ptr<GRBModel> &model, std::vector<GRBVar> &vars);

    template<class T>
    double maximizeObjectiveForEdge(int edge, int vert, std::function<T(GRBLinExpr, GRBLinExpr, GRBLinExpr)> obj);

    double maximizeObjectiveForEdge(int edge, int vert, VertexLinObjFunctor obj) { 
        return maximizeObjectiveForEdge<GRBLinExpr>(edge, vert, obj);
    };
    double maximizeObjectiveForEdge(int edge, int vert, VertexQuadObjFunctor obj) { 
        return maximizeObjectiveForEdge<GRBQuadExpr>(edge, vert, obj);
    };


};
double solveUsingL2NormSq(Eigen::MatrixXd C, Eigen::VectorXd b, Eigen::MatrixXd V, Eigen::MatrixXi E, bool verbose);
double solveUsingL1Norm(Eigen::MatrixXd C, Eigen::VectorXd b, Eigen::MatrixXd V, Eigen::MatrixXi E, bool verbose);

#endif
