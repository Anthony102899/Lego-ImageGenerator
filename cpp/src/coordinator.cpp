#include <memory>
#include <functional>
#include <Eigen>

#include "coordinator.h"
#include "gurobi_solver.h"


std::vector<ObjSolPair> solveVertexwiseUsingL1Norm(GurobiSolver solver, double eps, double maxCost) {
    try {
        solver.addConstraints(eps, maxCost);

        int numProblem = solver.C.cols() / 6;

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


        std::vector<ObjSolPair> results;

        for (int i = 0; i < numProblem; i++) {
            std::vector<ObjSolPair> objs0(8);
            std::vector<ObjSolPair> objs1(8);

            for (auto maker: linMakers) {
                double o0 = solver.maximizeObjectiveForEdge(i, 0, maker);
                double o1 = solver.maximizeObjectiveForEdge(i, 1, maker);
                std::vector<double> sol0 = solver.solution();
                std::vector<double> sol1 = solver.solution();
                objs0.emplace_back(o0, sol0);
                objs1.emplace_back(o1, sol1);
            }

            auto maxByFirst = [](ObjSolPair p0, ObjSolPair p1) { return p0.first < p1.first; };
            ObjSolPair maxPair0 = *std::max_element(objs0.begin(), objs0.end(), maxByFirst);
            ObjSolPair maxPair1 = *std::max_element(objs1.begin(), objs1.end(), maxByFirst);

            results.push_back(maxPair0);
            results.push_back(maxPair1);
        }

        return results;

    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw e;
    }
}

std::vector<ObjSolPair> solveVertexwiseUsingL2NormSq(GurobiSolver solver, double epsilon, double maxCost) {
    try {
        solver.addConstraints(epsilon, maxCost);

        int numProblem = solver.C.cols() / 6;

        VertexQuadObjFunctor maker = [](auto ux, auto uy, auto uz) {
            return ux * ux + uy * uy + uz * uz;
        };

        std::vector<ObjSolPair> results;

        solver.model->set(GRB_IntParam_NonConvex, 2);

        for (int i = 0; i < numProblem; i++) {
            double maxObj0 = solver.maximizeObjectiveForEdge(i, 0, maker);
            std::vector<double> sol0 = solver.solution();
            results.emplace_back(maxObj0, sol0);

            double maxObj1 = solver.maximizeObjectiveForEdge(i, 1, maker);
            std::vector<double> sol1 = solver.solution();
            results.emplace_back(maxObj1, sol1);
        }
        return results;
    } catch (GRBException e) {
        std::cout << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw e;
    }
}

double solveGlobally(GurobiSolver solver, double maxCost) {
    auto compute_length = [](Eigen::Vector3d p, Eigen::Vector3d q) {
        return (p - q).squaredNorm();
    };

    GRBQuadExpr energy;
    auto &vars = solver.vars;
    try {
        for (unsigned i = 0; i < vars.size(); i += 6) {
            int edgeInd = i / 6;
            std::cout << i << std::endl;
            double mass = 1;
            double length = compute_length(
                solver.V.row(solver.E(edgeInd, 0)),
                solver.V.row(solver.E(edgeInd, 1))
            );
            double inertia = (1.0 / 12.0) * mass * length;

            GRBQuadExpr vSq = (
                vars[i + 0] * vars[i + 0] + 
                vars[i + 1] * vars[i + 1] + 
                vars[i + 2] * vars[i + 2]);
            GRBQuadExpr wSq = 
                vars[i + 3] * vars[i + 3] + 
                vars[i + 4] * vars[i + 4] + 
                vars[i + 5] * vars[i + 5];

            GRBQuadExpr vEnergy = 0.5 * mass    * vSq;
            GRBQuadExpr wEnergy = 0.5 * inertia * wSq;

            energy += vEnergy;
            energy += wEnergy;
        }

        // solver.model->addQConstr(energy, GRB_LESS_EQUAL, maxCost);
        solver.model->addQConstr(energy, GRB_LESS_EQUAL, maxCost);

        // obj: MAXIMIZE f(s) = C.transpose() * C 
        std::vector<GRBLinExpr> exprs = solver.buildConstraintExpressions();
        GRBQuadExpr objective;

        for (auto &expr : exprs) {
            objective += expr * expr;
        }

        solver.model->setObjective(objective, GRB_MAXIMIZE);
        solver.model->set(GRB_IntParam_NonConvex, 2);
        solver.model->optimize();
        double obj = solver.objectiveValue();
        auto sol = solver.solution();

        std::cout << obj;
        for (auto &v : sol) {
            std::cout << v << " ";
        }

        return obj;
    } catch (GRBException e) {
        std::cerr << e.getErrorCode() << " " << e.getMessage() << std::endl;
        throw e;
    }
}
