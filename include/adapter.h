#include <sqlite3.h>
#include <string>
#include <memory>

#ifndef ADAPTER_H
#define APAPTER_H
struct SqliteAdapter {

    sqlite3 *db;
    int exit;
    char *errMsg;

    SqliteAdapter(std::string filename);
    ~SqliteAdapter();
    void initOptimizationTables();
    int writeOptimizationResult(
        std::string filename, 
        double eps,
        double maxCost,
        double objVal
    );

};
#endif