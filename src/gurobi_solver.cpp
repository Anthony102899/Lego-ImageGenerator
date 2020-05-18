#include <string>

#include "gurobi_solver.h"
#include "gurobi_c++.h"

using namespace Eigen;

// declaration
std::string varname(int n);

double solve_by_gurobi(MatrixXd C, VectorXd b, MatrixXd V, MatrixXi E, bool verbose) {
    // double *constraints = C.transpose().data();
    // double constraints_size = C.size();
    int num_vars = C.cols();
    int num_cons = C.rows();
    int num_probs = num_vars / 6 * 2;

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    model.set(GRB_IntParam_LogToConsole, (int) verbose);

    // decision variables
    try {
        GRBVar *vars = model.addVars(num_vars, GRB_CONTINUOUS);
        for (int i = 0; i < num_vars; i++) {
            vars[i].set(GRB_StringAttr_VarName, varname(i));
            // if (i < 6) {
            //     vars[i].set(GRB_DoubleAttr_UB, 0.0);
            //     vars[i].set(GRB_DoubleAttr_LB, 0.0);
            // } else {
            //     vars[i].set(GRB_DoubleAttr_UB, 10.0);
            //     vars[i].set(GRB_DoubleAttr_LB, -10.0);
            // }
            vars[i].set(GRB_DoubleAttr_UB, 10.0);
            vars[i].set(GRB_DoubleAttr_LB, -10.0);
        }
        std::cout << "variables initialized" << std::endl;

    // constraints
        double *coeffs = new double[num_vars];
        for (int i = 0; i < num_cons; i++) {
            GRBLinExpr expr;
            for (int j = 0; j < num_vars; j++) {
                expr += C(i, j) * vars[j];
            }
            std::string name = "cstr " + std::to_string(i);
            double target = b(i);
            if (i >= num_cons - 6) {
                model.addConstr(expr, GRB_LESS_EQUAL   , target, name);
                model.addConstr(expr, GRB_GREATER_EQUAL, target, name);
            } else {
                model.addConstr(expr, GRB_LESS_EQUAL   , target + 0.05, name);
                model.addConstr(expr, GRB_GREATER_EQUAL, target - 0.05, name);
            }
        }
        delete[] coeffs;

        double max = -1;
        std::vector<double> maxima;
        for (int i = 0; i < num_probs; i++) {
            int edge_ind = i / 2;
            Vector3d pt[2] = {
                V.row(E(edge_ind, 0)),
                V.row(E(edge_ind, 1))
            };
            Vector3d mid = (pt[0] + pt[1]) / 2;
            Vector3d a = pt[i % 2] - mid;

            // objective function
            const int8_t x = 0, y = 1, z = 2;
            int vx = edge_ind * 6 + 0;
            int vy = edge_ind * 6 + 1;
            int vz = edge_ind * 6 + 2;
            int wx = edge_ind * 6 + 3;
            int wy = edge_ind * 6 + 4;
            int wz = edge_ind * 6 + 5;
            GRBLinExpr ux = vars[vx] + a(z) * vars[wy] - a(y) * vars[wz];
            GRBLinExpr uy = vars[vy] + a(x) * vars[wz] - a(z) * vars[wx];
            GRBLinExpr uz = vars[vz] + a(y) * vars[wx] - a(x) * vars[wy];
            GRBQuadExpr objective = ux * ux + uy * uy + uz * uz;

            model.setObjective(objective, GRB_MAXIMIZE);

            // maximizing the velocity is non-convex
            model.set(GRB_IntParam_NonConvex, 2);

            model.optimize();

            double obj_val = model.get(GRB_DoubleAttr_ObjVal);
            for (int i = 0; i < num_vars; i++) {
                auto x = vars[i];
                std::cout << x.get(GRB_StringAttr_VarName) << " " << x.get(GRB_DoubleAttr_X);
                std::cout << (i % 6 == 5 ? "\n" : " ");
            }
            std::cout << "C DoF " << (C.cols() - C.fullPivHouseholderQr().rank()) << std::endl;
            std::cout << "Objective: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

            maxima.push_back(obj_val);
            max = (max < obj_val) ? obj_val : max;
        }

        for (double val: maxima) std::cout << val << " ";
        return max;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return -1;
    }
}

std::string varname(int n) {
    int i = n / 6;
    char vw = (n % 6 / 2 == 0) ? 'v' : 'w';
    char map[4] = {'x', 'y', 'z'};
    char vw_n = map[(n % 3)];
    return std::string(1, vw) + std::to_string(i) + std::string(1, vw_n);
}