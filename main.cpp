#include <Eigen>
#include <iostream>

#include "reader.h"
#include "solver.h"
#include "shifter.h"
#include "writer.h"

#define o(x) {std::cout << (x) << std::endl;}
#define oo(x, y) {std::cout << (x) << " " << (y) << std::endl;};

using namespace Eigen;

int main(int argc, char *argv[]) {
    MatrixXd P;
    MatrixXi E;
    MatrixXi pins;

    std::string data_file;
    if (argc < 2) {
        data_file = "data/sqaure.txt";
    } else {
        data_file = argv[1];
    }

    read_data_file(data_file.c_str(), P, E, pins);
    // trapezoid_data(P, E, pins);
    // square_data(P, E, pins);
    std::vector<MatrixXd> matrices;
    matrices.push_back(P);
    std::vector<std::pair<int, VectorXd>> unstable_indices;
    int dof;
    bool stable = solve(P, E, pins, dof, unstable_indices);
    oo("Degree of freedom:", dof);

    std::string output_filename = data_file.append(".out");

    if (stable) {

    } else { o("Unstable");
        double step = 0.00002;
        int iter_num = 1000;

        int significant_index = unstable_indices.at(0).first;
        VectorXd s = unstable_indices.at(0).second;

        for (int i = 0; i < iter_num; i++) {

            P = shift(P, E, s, step);

            if (i % 30 == 0) {
                o(i); 
                matrices.push_back(P);
            }

            std::vector<std::pair<int, VectorXd>> unstable_indices;
            int dof;
            bool stable = solve(P, E, pins, dof, unstable_indices);
            if (stable) {
                oo(i, "It's stable now!!");
                oo("now dof:", dof);
                break;
            } else {
                for (int i = 0; i < unstable_indices.size(); i++) {
                    if (unstable_indices.at(i).first == significant_index) {
                        s = unstable_indices.at(i).second;
                        break;
                    }
                }
            }
        }

    }

    write_matrices(output_filename.c_str(), matrices);

    return 0;
}