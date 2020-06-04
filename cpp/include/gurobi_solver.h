#include <Eigen>
#include <memory>
#include <functional>

#include "gurobi_c++.h"

#ifndef LP_SOLVER_H
#define LP_SOLVER_H

using VertexLinObjFunctor = std::function<GRBLinExpr(GRBLinExpr, GRBLinExpr, GRBLinExpr)>;
using VertexQuadObjFunctor = std::function<GRBQuadExpr(GRBLinExpr, GRBLinExpr, GRBLinExpr)>;

using ObjSolPair = std::pair<double, std::vector<double>>;

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
        initModelAndVars();
    };

    void initModelAndVars();
    bool addConstraints(double eps, double cost);

    std::vector<GRBLinExpr> buildConstraintExpressions();

    template<class T>
    double maximizeObjectiveForEdge(int edge, int vert, 
        std::function<T(GRBLinExpr, GRBLinExpr, GRBLinExpr)> makeObj) {
        using namespace Eigen;
        assert(vert == 0 || vert == 1);
        Vector3d pt[2] = {
            V.row(E(edge, 0)),
            V.row(E(edge, 1))
        };
        Vector3d mid = (pt[0] + pt[1]) / 2;
        Vector3d a = pt[vert] - mid;
        const int8_t x = 0, y = 1, z = 2;
        int vx = edge * 6 + 0; int vy = edge * 6 + 1; int vz = edge * 6 + 2;
        int wx = edge * 6 + 3; int wy = edge * 6 + 4; int wz = edge * 6 + 5;
        GRBLinExpr ux = vars[vx] + a(z) * vars[wy] - a(y) * vars[wz];
        GRBLinExpr uy = vars[vy] + a(x) * vars[wz] - a(z) * vars[wx];
        GRBLinExpr uz = vars[vz] + a(y) * vars[wx] - a(x) * vars[wy];

        T obj = makeObj(ux, uy, uz);
        model->setObjective(obj, GRB_MAXIMIZE);
        model->optimize();
        
        double objVal = model->get(GRB_DoubleAttr_ObjVal);
        return objVal;
    }

    double maximizeObjectiveForEdge(int edge, int vert, VertexLinObjFunctor obj) { 
        return maximizeObjectiveForEdge<GRBLinExpr>(edge, vert, obj);
    };
    double maximizeObjectiveForEdge(int edge, int vert, VertexQuadObjFunctor obj) { 
        return maximizeObjectiveForEdge<GRBQuadExpr>(edge, vert, obj);
    };

    std::vector<double> solution();
    double objectiveValue() { return model->get(GRB_DoubleAttr_ObjVal); };
};

std::string varname(int n);
#endif
