#include "reader.h"

using namespace Eigen;

bool read_data_file(const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins,
    Eigen::MatrixXi &anchors
) {
    FILE *fp = fopen(filename, "r");

    char str[10];
    int row_num;
    printf("Reading %s...\n", filename);

    #define read_section(M, type, template, row_num, col_num)\
    printf("Reading %d rows, %d cols into '%s'\n", row_num, col_num, str);\
    M.resize(row_num, col_num);\
    for (int i = 0; i < row_num; i++) {\
        for (int j = 0; j < col_num; j++) {\
            type reads;\
            if (fscanf(fp, template, &reads) == EOF) {\
                fprintf(stderr, "Unexpected EOF, check if anything wrong with the data file %s\n", filename);\
            }\
            M(i, j) = reads;\
        }\
    }

    if (
        fscanf(fp, "%s", str) != EOF && 
        fscanf(fp, "%d", &row_num) != EOF
    ) {
        read_section(P, double, "%lf", row_num, 3);
    }
    
    if (
        fscanf(fp, "%s", str) != EOF && 
        fscanf(fp, "%d", &row_num) != EOF
    ) {
        read_section(E, int, "%d", row_num, 2);
    }

    if (
        fscanf(fp, "%s", str) != EOF &&
        fscanf(fp, "%d", &row_num) != EOF 
    ) {
        read_section(pins, int, "%d", row_num, 3);
    }

    if (
        fscanf(fp, "%s", str) != EOF &&
        fscanf(fp, "%d", &row_num) != EOF 
    ) {
        read_section(anchors, int, "%d", row_num, 3);
    }


    printf("Finished reading data\n");
    fclose(fp);

    return true;
}