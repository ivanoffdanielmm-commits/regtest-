#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

inline double norm2(const Vector& v) {
    double sum = 0.0;
    for (double val : v) sum += val * val;
    return std::sqrt(sum);
}