#include <memory>
#include <functional>

#include "coordinator.h"

std::vector<ObjSolPair> solveUsingL1Norm(GurobiSolver solver, double eps, double maxCost) {
    try {
        solver.buildConstraints(eps, maxCost);

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

std::vector<ObjSolPair> solveUsingL2NormSq(GurobiSolver solver, double epsilon, double maxCost) {
    try {
        solver.buildConstraints(epsilon, maxCost);

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
