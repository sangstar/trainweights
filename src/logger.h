//
// Created by Sanger Steel on 2/10/26.
//

#ifndef TRAINWEIGHTS_LOGGER_H
#define TRAINWEIGHTS_LOGGER_H

#include <iostream>

enum class LoggingLevels {
    None,
    Info,
    Debug
};

struct Logger {
    enum LoggingLevels level;
    template<typename... Args>
    static void log(Args&&... args) {
        (std::cout << ... << args) << '\n';
    }
    template<typename... Args>
    void debug(Args&&... args) {
        if (level == LoggingLevels::Debug) {
            (std::cout << ... << args) << '\n';
        }
    }

    template<typename... Args>
    void debugf(const char* msg, Args&&... args) {
        printf(msg, args...);
        putchar('\n');
    }

    Logger() : level(LoggingLevels::None) {};
    Logger(LoggingLevels level) : level(level) {};
};

extern Logger logger;

#endif //TRAINWEIGHTS_LOGGER_H