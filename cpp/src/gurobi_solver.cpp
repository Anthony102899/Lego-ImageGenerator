#include <string>
#include <functional>
#include <memory>
#include <algorithm>
#include <utility>

#include "gurobi_solver.h"

using namespace Eigen;

// declaration
std::string varname(int n);

// implentation
std::vector<double> GurobiSolver::solution() {
    std::vector<double> sol;
    auto getValue = [](GRBVar v) { return v.get(GRB_DoubleAttr_X); };
    std::transform(vars.begin(), vars.end(), std::back_inserter(sol), getValue);
    return sol;
}

void GurobiSolver::initModelAndVars() {
    GRBEnv env = GRBEnv();
    model = std::make_shared<GRBModel>(env);
    model->set(GRB_IntParam_LogToConsole, (int) verbose);
    // decision variables
    try {
        int numVars = C.cols();
        double *lb = new double[numVars];
        double *ub = new double[numVars];
        double *obj = NULL;
        char   *type = new char[numVars];
        std::string *name = new std::string[numVars];

        for (int i = 0; i < numVars; i++) {
            ub[i] = 10.0;
            lb[i] = -10.0;
            type[i] = GRB_CONTINUOUS;
            name[i] = varname(i);
        }
        // GRBVar *varArray = model->addVars(numVars, GRB_CONTINUOUS);
        // std::vector<GRBVar> vars;
        GRBVar *varArray = model->addVars(lb, ub, obj, type, name, numVars);
        for (int i = 0; i < numVars; i++) {
            vars.push_back(varArray[i]);
        }
    } catch (GRBException e) {
        std::cerr << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw e;
    }
}

std::vector<GRBLinExpr> GurobiSolver::buildConstraintExpressions() {
    std::vector<GRBLinExpr> exprs;
    int numCons = C.rows();
    int numVars = C.cols();

    for (unsigned i = 0; i < numCons; i++) {
        GRBLinExpr expr;
        for (int j = 0; j < numVars; j++) {
            expr += C(i, j) * vars.at(j);
        }
        exprs.push_back(expr);
    }

    return exprs;
}


bool GurobiSolver::addConstraints(double eps, double maxCost) {
    assert(maxCost >= 0);

    int numVars = C.cols();
    int numCons = C.rows();
    GRBEnv env = GRBEnv();
    try {   
        model->set(GRB_IntParam_LogToConsole, (int) verbose);
        // constraints
        GRBQuadExpr overallCost;
        for (unsigned i = 0; i < numCons; i++) {
            GRBLinExpr expr;

            for (unsigned j = 0; j < numVars; j++) {
                expr += C(i, j) * vars[j];
            }
            std::string name = "cstr " + std::to_string(i);
            double target = b(i);

            // translation. No error allowed
            // if (i % 6 == 0 || i % 6 == 1 || i % 6 == 2) {
            //     model->addConstr(expr, GRB_LESS_EQUAL   , target, name);
            //     model->addConstr(expr, GRB_GREATER_EQUAL, target, name);
            //     continue;
            // }

            GRBQuadExpr square = expr * expr - target;

            if (i >= numCons - 6) {
                model->addConstr(expr, GRB_LESS_EQUAL   , target, name);
                model->addConstr(expr, GRB_GREATER_EQUAL, target, name);
            } else {
                model->addQConstr(square, GRB_LESS_EQUAL   , target + eps, name + "square");
                overallCost += square;
            }
        }

        model->addQConstr(overallCost <= maxCost, "cost");

        return true;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return false;
    }
}

std::string varname(int n) {
    int i = n / 6;
    char vw = (n % 6  < 3) ? 'v' : 'w';
    char map[4] = {'x', 'y', 'z'};
    char vw_n = map[(n % 3)];
    return std::string(1, vw) + std::to_string(i) + std::string(1, vw_n);
}