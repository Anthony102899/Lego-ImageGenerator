#include "reader.h"

using namespace Eigen;

bool read_data_file(const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins) 
{
    FILE *fp = fopen(filename, "r");

    char *str;
    int row_num;
    int col_num;
    fscanf(fp, "%s", str);
    fscanf(fp, "%d", &row_num);
    col_num = 3;
    P.resize(row_num, col_num);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            float reads;
            fscanf(fp, "%f", &reads);
            P(i, j) = reads;
        }
    }
    fscanf(fp, "%s", str);
    fscanf(fp, "%d", &row_num);
    col_num = 2;
    E.resize(row_num, col_num);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            int reads;
            fscanf(fp, "%d", &reads);
            E(i, j) = reads;
        }
    }
    fscanf(fp, "%s", str);
    fscanf(fp, "%d", &row_num);
    col_num = 3;
    pins.resize(row_num, col_num);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            int reads;
            fscanf(fp, "%d", &reads);
            pins(i, j) = reads;
        }
    }
    fclose(fp);
}