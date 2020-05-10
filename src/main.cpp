#include <Eigen>
#include <iostream>
#include <cstdio>

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
    MatrixXi anchors;

    std::string data_file;
    if (argc < 2) {
        data_file = "data/sqaure.txt";
    } else {
        data_file = argv[1];
    }

    read_data_file(data_file.c_str(), P, E, pins, anchors);
    std::string output_filename = data_file + ".out";
    std::string direction_filename = data_file + ".drt.out";

    std::vector<MatrixXd> hist;
    std::vector<MatrixXd> init_direction;
    hist.push_back(P);
    std::vector<std::tuple<int, VectorXd, double>> unstable_indices;

    int dof; MatrixXd C;
    bool stable = solve(P, E, pins, anchors, dof, C, unstable_indices);

    if (!stable) {
        for (unsigned i = 0; i < unstable_indices.size(); i++) {
            VectorXd velocity = std::get<1>(unstable_indices[i]);
            auto P_new = shift(P, E, velocity, 0.001);
            MatrixXd D = P_new - P;
            init_direction.push_back(D);
        }
        write_matrices(direction_filename.c_str(), init_direction);
    }

    if (stable) {
        o("Stable");
    } else { 
        o("Unstable");
        oo("Degree of freedom:", dof);
        double step = 2e-5;
        int iter_num = 1000;

        int significant_index; VectorXd s; double error;
        std::tie(significant_index, s, error) = unstable_indices.at(0);
        oo("Initial speed", s.transpose());

        for (int i = 0; i < iter_num; i++) {

            auto P_new = shift(P, E, s, step);
            MatrixXd D = P_new - P;
            P = P_new;
            if (i == 0) {
                o(D);
            }

            if (i % 20 == 0 || i < 3) {
                oo("iter:", i); 
                hist.push_back(P);
                // Vector3d mid0 = (P.row(E(1, 0)) + P.row(E(1, 1))) / 2.0;
                // Vector3d mid2 = (P.row(E(3, 0)) + P.row(E(3, 1))) / 2.0;
                // double length = (mid0 - mid2).norm();
                // for (int e = 0; e < E.rows(); e++) {
                //     Vector3d p1 = P.row(E(e, 0));
                //     Vector3d p2 = P.row(E(e, 1));
                //     double n = (p1 - p2).norm();
                //     oo(e, n);
                // }
                // oo("mid-length", length);
            }

            std::vector<std::tuple<int, VectorXd, double>> unstable_indices;
            int dof;
            bool stable = solve(P, E, pins, anchors, dof, C, unstable_indices);
            if (stable) {
                oo(i, "It's stable now!!");
                oo("now dof:", dof);
                // FILE * f = fopen("temp.csv", "w");
                // for (int i = 0; i < C.rows(); i++) {
                //     for (int j = 0; j < C.cols(); j++) {
                //         fprintf(f, "%10.7f", C(i, j));
                //         if (j != C.cols() - 1)
                //         fprintf(f, ",");
                //     }
                //     fprintf(f, "\n");
                // }
                // fclose(f);
                // o("End position");

                for (int e = 0; e < E.rows(); e++) {
                    Vector3d p1 = P.row(E(e, 0));
                    Vector3d p2 = P.row(E(e, 1));
                    double n = (p1 - p2).norm();
                    oo(e, n);
                }
                o(P);
                break;
            } else {
                for (unsigned i = 0; i < unstable_indices.size(); i++) {
                    if (std::get<0>(unstable_indices.at(i)) == significant_index) {
                        s = std::get<1>(unstable_indices.at(i));
                        error = std::get<2>(unstable_indices.at(i));
                        break;
                    }
                }
            }
        }

    }

    write_matrices(output_filename.c_str(), hist);

    return 0;
}