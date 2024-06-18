//
// Created by nn on 4/5/24.
//

#ifndef IMMVISION_DATABASE_H
#define IMMVISION_DATABASE_H


#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "appstate.h"

#include <chrono>
#include <iomanip>
#include <sstream>

std::string getCurrentDate() {
    // Get current time as time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t for use with gmtime
    auto nowAsTimeT = std::chrono::system_clock::to_time_t(now);

    // Convert to tm struct
    std::tm nowTm = *std::gmtime(&nowAsTimeT);

    // Subtract to get the number of milliseconds
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // Create a stringstream to format the output
    std::stringstream ss;
    ss << std::put_time(&nowTm, "%Y-%m-%d %H:%M:%S.") << std::setfill('0') << std::setw(3) << milliseconds.count();

    return ss.str();
}


class DatabaseHandler {
public:
    DatabaseHandler(const std::string& filename) : db(filename, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE) {
        std::cout << "SQLite database file '" << db.getFilename().c_str() << "' opened successfully\n";
        createTables();
    }

    void createTables() {
        try {
            db.exec("CREATE TABLE IF NOT EXISTS json_data (date TEXT, json_string TEXT)");
            // Other table creations...
        } catch (std::exception& e) {
            std::cout << "SQLite exception: " << e.what() << std::endl;
            throw;
        }
    }

    void insertJsonData(const std::string& date, const std::string& json_string) {
        try {
            SQLite::Statement query(db, "INSERT INTO json_data VALUES (?, ?)");
            query.bind(1, date);
            query.bind(2, json_string);
            query.exec();
        } catch (std::exception& e) {
            std::cout << "SQLite exception: " << e.what() << std::endl;
            throw;
        }
    }

    void deleteJsonData(const std::string& date) {
        try {
            SQLite::Statement query(db, "DELETE FROM json_data WHERE date = ?");
            query.bind(1, date);
            int rowsDeleted = query.exec();
            std::cout << "Deleted " << rowsDeleted << " rows.\n";
        } catch (std::exception& e) {
            std::cout << "SQLite exception: " << e.what() << std::endl;
            throw;
        }
    }

private:
    SQLite::Database db;
};
#endif //IMMVISION_DATABASE_H
