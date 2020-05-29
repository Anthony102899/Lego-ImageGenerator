#include <cstdio>
#include <fstream>
#include "writer.h"


void writeMatrices(const char *filename, std::vector<Eigen::MatrixXd> matrices) {
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

json parseObjSolPairsToJson(std::vector<ObjSolPair> pairs) {
    json j;
    for (unsigned i = 0; i < pairs.size(); i++) {
        auto &p = pairs[i];

        json jp;
        jp["edge"] = i / 2;
        jp["vertex"] = i % 2;
        jp["obj"] = p.first;
        jp["sol"] = json(p.second);

        j.push_back(jp);
    }
    return j;
}

void writeJsonToFile(std::string filename, json j) {
    std::ofstream fo(filename);
    if (fo.is_open()){
        fo << j;
        fo.close();
    } else {
        std::cerr << "something went wrong when opening the json file " 
                  << filename 
                  << " for writing" << std::endl;
    }
}

void writeObjSolPairsJson(std::string filename, std::vector<ObjSolPair> pairs) {
    json j = parseObjSolPairsToJson(pairs);
    writeJsonToFile(filename, j);
}