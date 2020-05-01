#include <cstdio>
#include "writer.h"

void write_matrices(const char *filename, std::vector<Eigen::MatrixXd> matrices) {
    FILE *fp = fopen(filename, "w");
    int length = matrices.size();

    fprintf(fp, "%d\n", length);
    for (int i = 0; i < length; i++) {
        Eigen::MatrixXd mat = matrices.at(i);
        int rows = mat.rows();
        int cols = mat.cols();
        fprintf(fp, "%d %d\n", rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                fprintf(fp, "%f ", mat(r, c));
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}