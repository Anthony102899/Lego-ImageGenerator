#include <cstdio>
#include <fstream>
#include "writer.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

json parseObjSolPairsToJson(std::vector<ObjSolPair> pairs);
void writeJsonToFile(std::string filename, json j);

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

void writeObjSolPairs2dJson(
    std::string filename,
    std::vector<double> epsilons,
    std::vector<double> costs, 
    std::vector<std::vector<ObjSolPair>> pairs2d) 
{
    json j;
    assert(epsilons.size() == costs.size() && costs.size() == pairs2d.size());

    for (unsigned i = 0; i < epsilons.size(); i++) {
        auto &pairs = pairs2d[i];
        json item;
        item["data"] = parseObjSolPairsToJson(pairs);
        item["eps"] = epsilons[i];
        item["cost"] = costs[i];

        j.push_back(item);
    }
    
    writeJsonToFile(filename, j);
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
        std::cerr << "something went wrong when opening the json file for writing" << std::endl;
    }
}

void writeObjSolPairsJson(std::string filename, std::vector<ObjSolPair> pairs) {
    json j = parseObjSolPairsToJson(pairs);
    writeJsonToFile(filename, j);
}