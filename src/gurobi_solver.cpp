#include <string>
#include <functional>
#include <memory>
#include <algorithm>

#include "gurobi_solver.h"

using namespace Eigen;

// declaration
std::string varname(int n);

// implentation
bool GurobiSolver::buildModel(std::shared_ptr<GRBModel> &model, std::vector<GRBVar> &vars) {
    int num_vars = C.cols();
    int num_cons = C.rows();

    GRBEnv env = GRBEnv();
    try {   
        model = std::make_shared<GRBModel>(env);
        model->set(GRB_IntParam_LogToConsole, (int) verbose);
        // decision variables
        std::cout << num_vars << "-" << std::endl;
        GRBVar *var_array = model->addVars(num_vars, GRB_CONTINUOUS);
        // std::vector<GRBVar> vars;
        for (int i = 0; i < num_vars; i++) {
            var_array[i].set(GRB_StringAttr_VarName, varname(i));
            var_array[i].set(GRB_DoubleAttr_UB, 10.0);
            var_array[i].set(GRB_DoubleAttr_LB, -10.0);

            vars.push_back(var_array[i]);
        }
        // constraints
        for (int i = 0; i < num_cons; i++) {
            GRBLinExpr expr;
            for (int j = 0; j < num_vars; j++) {
                expr += C(i, j) * vars[j];
            }
            std::string name = "cstr " + std::to_string(i);
            double target = b(i);
            if (i >= num_cons - 6) {
                model->addConstr(expr, GRB_LESS_EQUAL   , target, name);
                model->addConstr(expr, GRB_GREATER_EQUAL, target, name);
            } else {
                model->addConstr(expr, GRB_LESS_EQUAL   , target + 0.05, name);
                model->addConstr(expr, GRB_GREATER_EQUAL, target - 0.05, name);
            }
        }
        return true;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return false;
    }
}


template<class T>
double GurobiSolver::maximizeObjectiveForEdge(int edge, int vert, 
    std::function<T(GRBLinExpr, GRBLinExpr, GRBLinExpr)> makeObj) {
    assert(vert == 0 || vert == 1);
    Vector3d pt[2] = {
        V.row(E(edge, 0)),
        V.row(E(edge, 1))
    };
    Vector3d mid = (pt[0] + pt[1]) / 2;
    Vector3d a = pt[vert] - mid;
    const int8_t x = 0, y = 1, z = 2;
    int vx = edge * 6 + 0;
    int vy = edge * 6 + 1;
    int vz = edge * 6 + 2;
    int wx = edge * 6 + 3;
    int wy = edge * 6 + 4;
    int wz = edge * 6 + 5;
    GRBLinExpr ux = vars[vx] + a(z) * vars[wy] - a(y) * vars[wz];
    GRBLinExpr uy = vars[vy] + a(x) * vars[wz] - a(z) * vars[wx];
    GRBLinExpr uz = vars[vz] + a(y) * vars[wx] - a(x) * vars[wy];
    T obj = makeObj(ux, uy, uz);
    model->setObjective(obj, GRB_MAXIMIZE);
    model->optimize();
    
    double objVal = model->get(GRB_DoubleAttr_ObjVal);
    return objVal;
}

double solveUsingL1Norm(MatrixXd C, VectorXd b, MatrixXd V, MatrixXi E, bool verbose) {
    try {
        GurobiSolver solver(C, b, V, E, verbose);
        int num_probs = C.cols() / 6;

        std::vector<VertexLinObjFunctor> linMakers {
            [](auto ux, auto uy, auto uz) { return  ux + uy + uz; },
            [](auto ux, auto uy, auto uz) { return  ux + uy - uz; },
            [](auto ux, auto uy, auto uz) { return  ux - uy + uz; },
            [](auto ux, auto uy, auto uz) { return  ux - uy - uz; },
            [](auto ux, auto uy, auto uz) { return -ux + uy + uz; },
            [](auto ux, auto uy, auto uz) { return -ux + uy - uz; },
            [](auto ux, auto uy, auto uz) { return -ux - uy + uz; },
            [](auto ux, auto uy, auto uz) { return -ux - uy - uz; }
        };

        for (int i = 0; i < num_probs; i++) {
            std::vector<double> objs0(8);
            std::vector<double> objs1(8);
            for (auto maker: linMakers) {
                double o0 = solver.maximizeObjectiveForEdge(i, 0, maker);
                double o1 = solver.maximizeObjectiveForEdge(i, 1, maker);
                objs0.push_back(o0);
                objs1.push_back(o1);
            }
            double maxObj0 = *std::max_element(objs0.begin(), objs0.end());
            double maxObj1 = *std::max_element(objs1.begin(), objs1.end());
            std::cout << "Edge " << i << " [0]: " << maxObj0 << " [1]: " << maxObj1 << std::endl;
        }
        return 0;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return -1;
    }
}

double solveUsingL2NormSq(MatrixXd C, VectorXd b, MatrixXd V, MatrixXi E, bool verbose) {
    try {
        GurobiSolver solver(C, b, V, E, verbose);
        int num_probs = C.cols() / 6;

        std::vector<VertexLinObjFunctor> linMakers {
            [](auto ux, auto uy, auto uz) { return  ux + uy + uz; },
            [](auto ux, auto uy, auto uz) { return  ux + uy - uz; },
            [](auto ux, auto uy, auto uz) { return  ux - uy + uz; },
            [](auto ux, auto uy, auto uz) { return  ux - uy - uz; },
            [](auto ux, auto uy, auto uz) { return -ux + uy + uz; },
            [](auto ux, auto uy, auto uz) { return -ux + uy - uz; },
            [](auto ux, auto uy, auto uz) { return -ux - uy + uz; },
            [](auto ux, auto uy, auto uz) { return -ux - uy - uz; }
        };

        for (int i = 0; i < num_probs; i++) {
            std::vector<double> objs0(8);
            std::vector<double> objs1(8);
            for (auto maker: linMakers) {
                double o0 = solver.maximizeObjectiveForEdge(i, 0, maker);
                double o1 = solver.maximizeObjectiveForEdge(i, 1, maker);
                objs0.push_back(o0);
                objs1.push_back(o1);
            }
            double maxObj0 = *std::max_element(objs0.begin(), objs0.end());
            double maxObj1 = *std::max_element(objs1.begin(), objs1.end());
        }

        for (int i = 0; i < num_probs; i++) {
            VertexQuadObjFunctor maker = [](auto ux, auto uy, auto uz) {
                return ux * ux + uy * uy + uz * uz;
            };
            double maxObj0 = solver.maximizeObjectiveForEdge(i, 0, maker);
            double maxObj1 = solver.maximizeObjectiveForEdge(i, 1, maker);
            solver.model->set(GRB_IntParam_NonConvex, 2);
            std::cout << "Edge " << i << " [0]: " << maxObj0 << " [1]: " << maxObj1 << std::endl;
        }
        return 0;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        return -1;
    }
}

std::string varname(int n) {
    int i = n / 6;
    char vw = (n % 6  < 3) ? 'v' : 'w';
    char map[4] = {'x', 'y', 'z'};
    char vw_n = map[(n % 3)];
    return std::string(1, vw) + std::to_string(i) + std::string(1, vw_n);
}