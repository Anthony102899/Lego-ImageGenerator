#include "reader.h"

using namespace Eigen;

template <class Ty>
bool read_one_section(FILE *fp, Eigen::Matrix<Ty, -1, -1> &M, const char read_template[], int row_num, int col_num) {
    M.resize(row_num, col_num);
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            Ty reads;
            if (fscanf(fp, read_template, &reads) == EOF) {
                fprintf(stderr, "Unexpected EOF at (%d,%d), check if anything wrong with the data file\n", i, j);
            }
            M(i, j) = reads;
        }
    }
    return true;
}

bool read_data_file(const char *filename, 
    Eigen::MatrixXd &P, 
    Eigen::MatrixXi &E, 
    Eigen::MatrixXi &pins,
    Eigen::MatrixXi &anchors
) {
    FILE *fp = fopen(filename, "r");

    char str[10];
    int row_num = 0;
    printf("Reading %s...\n", filename);

    while (
        fscanf(fp, "%s", str) != EOF && 
        fscanf(fp, "%d", &row_num) != EOF
    ) {
        printf("read %c, row %d\n", str[0], row_num);
        if (str[0] == 'P') read_one_section<double>(fp, P, "%lf", row_num, 3);
        if (str[0] == 'E') read_one_section<int>(fp, E, "%d", row_num, 2);
        if (str[0] == 'p') read_one_section<int>(fp, pins, "%d", row_num, 3);
        if (str[0] == 'a') read_one_section<int>(fp, anchors, "%d", row_num, 3);
    }


    printf("Finished reading data\n");
    printf("Matrix P (%d, %d), E (%d, %d), pins (%d, %d), anchors (%d, %d)\n",
        (int) P.rows(), (int) P.cols(),
        (int) E.rows(), (int) E.cols(),
        (int) pins.rows(), (int) pins.cols(),
        (int) anchors.rows(), (int) anchors.cols());
    fclose(fp);

    return true;
}