#include "gurobi_solver.h"

#ifndef COORDINATOR_H
std::vector<ObjSolPair> solveVertexwiseL2NormSq(GurobiSolver solver, double epsilon, double maxCost);
std::vector<ObjSolPair> solveVertexwiseL1Norm  (GurobiSolver solver, double epsilon, double maxCost);
double                  solveGlobally          (GurobiSolver solver, double maxCost);
#endif // !COORDINATOR_H