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
    std::vector<std::tuple<int, VectorXd, double>> unstable_indices;
    int dof; MatrixXd C;
    bool stable = solve(P, E, pins, dof, C, unstable_indices);
    oo("Degree of freedom:", dof);

    std::string output_filename = data_file.append(".out");

    if (stable) {
        o("Stable");
    } else { 
        o("Unstable");
        double step = 0.00002;
        int iter_num = 1000;

        int significant_index; VectorXd s; double error;
        std::tie(significant_index, s, error) = unstable_indices.at(0);
        oo("Initial speed", s.transpose());

        for (int i = 0; i < iter_num; i++) {

            P = shift(P, E, s, step);
            if (i <= 1) {
                auto D = displacement_matrix(P, E, s);
                o("Displacement matrix"); o(D);
                o(i);
                o("s");
                o(s.transpose());
                // o("constraint matrix");
                // auto block1 = C.block<5, 6>(4 * 5, 1 * 6);
                // auto block2 = C.block<5, 6>(4 * 5, 4 * 6);
                // o(block1); o(""); o(block2);
                oo("error", error);
                o("shifted Points");
                o(P);
            }

            if (i % 30 == 0 || i < 3) {
                oo("iter:", i); 
                matrices.push_back(P);
            }

            std::vector<std::tuple<int, VectorXd, double>> unstable_indices;
            int dof;
            bool stable = solve(P, E, pins, dof, C, unstable_indices);
            if (stable) {
                oo(i, "It's stable now!!");
                oo("now dof:", dof);
                o("End position");
                o(P);
                break;
            } else {
                for (int i = 0; i < unstable_indices.size(); i++) {
                    if (std::get<0>(unstable_indices.at(i)) == significant_index) {
                        s = std::get<1>(unstable_indices.at(i));
                        error = std::get<2>(unstable_indices.at(i));
                        break;
                    }
                }
            }
        }

    }

    write_matrices(output_filename.c_str(), matrices);

    return 0;
}