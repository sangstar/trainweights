//
// Created by Sanger Steel on 2/10/26.
//

#ifndef TRAINWEIGHTS_LOGGER_H
#define TRAINWEIGHTS_LOGGER_H

#include <iostream>

struct Logger {
    template<typename... Args>
    static void log(Args&&... args) {
        (std::cout << ... << args) << '\n';
    }
};

#endif //TRAINWEIGHTS_LOGGER_H