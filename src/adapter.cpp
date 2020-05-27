#include "adapter.h"

SqliteAdapter::SqliteAdapter(std::string filename) {
    exit = sqlite3_open(filename.c_str(), &db);
    if (exit != SQLITE_OK) {

    }
}

SqliteAdapter::~SqliteAdapter() {
    sqlite3_close(db);
}

void SqliteAdapter::initOptimizationTables() {
    std::string createSql = ""
    "CREATE TABLE IF NOT EXISTS "
    "file ("
    "   id INT PRIMARY KEY AUTO INCREMENT, "
    "   filename TEXT NOT NULL, "
    "   content TEXT  "
    ");"
    "CREATE TABLE IF NOT EXISTS "
    "opt_result ("
    "   id INT PRIMARY KEY AUTO INCREMENT, "
    "   epsilon DOUBLE NOT NULL, "
    "   max_cost DOUBLE NOT NULL, "
    "   obj_val REAL NOT NULL, "
    "   file_id INT NOT NULL, "
    "   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"
    ");"
    ;

    sqlite3_exec(db, createSql.c_str(), 0, 0, &errMsg);

}

int SqliteAdapter::writeOptimizationResult(
    std::string filename, 
    double eps,
    double maxCost,
    double objVal
) {
    std::string updateFilename = ""
    "INSERT INTO opt_result () VALUES (?);";
    std::string insertResult = ""
    "INSERT INTO opt_result (epsilon, max_cost, obj_val, file_id) VALUES (?, ?, ?, ?);" ;
    return 0;
}