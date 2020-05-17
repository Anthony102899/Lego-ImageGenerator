#include <string>

#include "gurobi_solver.h"
#include "gurobi_c++.h"

using namespace Eigen;

// declaration
std::string varname(int n);

int solve_by_gurobi(MatrixXd C, VectorXd b, MatrixXd V, MatrixXi E) {
    // double *constraints = C.transpose().data();
    // double constraints_size = C.size();
    int num_vars = C.cols();
    int num_cons = C.rows();
    int num_probs = num_vars / 6 * 2;

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // decision variables
    try {
        GRBVar *vars = model.addVars(num_vars, GRB_CONTINUOUS);
        for (int i = 0; i < num_vars; i++) {
            vars[i].set(GRB_StringAttr_VarName, varname(i));
            vars[i].set(GRB_DoubleAttr_UB, GRB_INFINITY);
            vars[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);
        }
        std::cout << "variables initialized" << std::endl;

    // constraints
        double *coeffs = new double[num_vars];
        for (int i = 0; i < num_cons; i++) {
            GRBLinExpr expr = C(i, 0) * vars[0];
            // std::cout << "&" << expr << std::endl;
            // for (int j = 1; j < num_vars; j++) {
            //     expr += C(i, j) * vars[j];
            //     // std::cout << "^" << expr << std::endl;
            // }
            // expr.addTerms(coeffs, variables, num_vars);
            std::string name = "cstr " + std::to_string(i);
            double target = b(i);
            model.addConstr(expr, GRB_LESS_EQUAL   , target + 0.05, name);
            model.addConstr(expr, GRB_GREATER_EQUAL, target - 0.05, name);
        }
        delete[] coeffs;

        int i = 0;
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

        model.setObjective(objective);

        model.optimize();

        std::cout << "Constraint Matrix" << std::endl << C << std::endl;

        double obj_val = model.get(GRB_DoubleAttr_ObjVal);
        for (int i = 0; i < num_vars; i++) {
            auto x = vars[i];
            std::cout << x.get(GRB_StringAttr_VarName) << " " << x.get(GRB_DoubleAttr_X);
            std::cout << (i % 6 == 5 ? "\n" : " ");
        }
        std::cout << "Objective: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

        return (int) obj_val;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return 0;
    }
}

std::string varname(int n) {
    int i = n / 6;
    char vw = (n % 6 / 2 == 0) ? 'v' : 'w';
    char map[4] = {'x', 'y', 'z'};
    char vw_n = map[(n % 3)];
    return std::string(1, vw) + std::to_string(i) + std::string(1, vw_n);
}