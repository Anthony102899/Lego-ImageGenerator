#include "reader.h"

using namespace Eigen;

bool read_data_file(const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins,
    Eigen::MatrixXi &anchors
) {
    FILE *fp = fopen(filename, "r");

    char *str;
    int row_num;
    int col_num;


    if (
        fscanf(fp, "%s", str) != EOF && 
        fscanf(fp, "%d", &row_num) != EOF
    ) {
        col_num = 3;
        printf("Reading %d rows, %d cols into '%s'\n", row_num, col_num, str);
        P.resize(row_num, col_num);
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < col_num; j++) {
                double reads;
                fscanf(fp, "%lf", &reads);
                P(i, j) = reads;
            }
        }
    }
    
    if (
        fscanf(fp, "%s", str) != EOF && 
        fscanf(fp, "%d", &row_num) != EOF
    ) {
        col_num = 2;
        printf("Reading %d rows, %d cols into '%s'\n", row_num, col_num, str);
        E.resize(row_num, col_num);
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < col_num; j++) {
                int reads;
                fscanf(fp, "%d", &reads);
                E(i, j) = reads;
            }
        }
    }

    if (
        fscanf(fp, "%s", str) != EOF &&
        fscanf(fp, "%d", &row_num) != EOF 
    ) {
        col_num = 3;
        printf("Reading %d rows, %d cols into '%s'\n", row_num, col_num, str);
        pins.resize(row_num, col_num);
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < col_num; j++) {
                int reads;
                fscanf(fp, "%d", &reads);
                pins(i, j) = reads;
            }
        }
    }

    if (
        fscanf(fp, "%s", str) != EOF &&
        fscanf(fp, "%d", &row_num) != EOF 
    ) {
        col_num = 3;
        printf("Reading %d rows, %d cols into '%s'\n", row_num, col_num, str);
        anchors.resize(row_num, col_num);
        for (int i = 0; i < row_num; i++) {
            for (int j = 0; j < col_num; j++) {
                int reads;
                fscanf(fp, "%d", &reads);
                anchors(i, j) = reads;
            }
        }
    }


    fclose(fp);
}