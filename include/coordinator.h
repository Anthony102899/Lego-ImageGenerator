#include "gurobi_solver.h"

#ifndef COORDINATOR_H
std::vector<ObjSolPair> solveUsingL2NormSq(GurobiSolver solver, double epsilon, double maxCost);
std::vector<ObjSolPair> solveUsingL1Norm  (GurobiSolver solver, double epsilon, double maxCost);
#endif // !COORDINATOR_H