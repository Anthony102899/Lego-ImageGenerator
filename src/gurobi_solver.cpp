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

bool GurobiSolver::buildConstraints(double eps, double maxCost) {
    assert(maxCost >= 0);

    int num_vars = C.cols();
    int num_cons = C.rows();

    GRBEnv env = GRBEnv();
    try {   
        model = std::make_shared<GRBModel>(env);
        model->set(GRB_IntParam_LogToConsole, (int) verbose);
        // decision variables
        GRBVar *var_array = model->addVars(num_vars, GRB_CONTINUOUS);
        // std::vector<GRBVar> vars;
        for (int i = 0; i < num_vars; i++) {
            var_array[i].set(GRB_StringAttr_VarName, varname(i));
            var_array[i].set(GRB_DoubleAttr_UB, 10.0);
            var_array[i].set(GRB_DoubleAttr_LB, -10.0);

            // 2d, set wx, wy, vz = 0
            if (i % 6 == 3 || i % 6 == 4 || i % 6 == 2) {
                var_array[i].set(GRB_DoubleAttr_UB, 0);
                var_array[i].set(GRB_DoubleAttr_LB, 0);
            }
            vars.push_back(var_array[i]);
        }
        // constraints
        GRBQuadExpr overallCost;
        for (int i = 0; i < num_cons; i++) {
            GRBLinExpr expr;

            for (int j = 0; j < num_vars; j++) {
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

            if (i >= num_cons - 6) {
                model->addConstr(expr, GRB_LESS_EQUAL   , target, name);
                model->addConstr(expr, GRB_GREATER_EQUAL, target, name);
            } else {
                model->addConstr(expr, GRB_LESS_EQUAL   , target + eps, name + "p");
                model->addConstr(expr, GRB_GREATER_EQUAL, target - eps, name + "n");
            }
            overallCost += expr * expr;
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