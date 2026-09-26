#pragma once
#include "matrix_ops.h"

struct Metrics {
    double eps_rel;
    double rho;
    double eta;
};

Metrics computeMetrics(const Vector& x_alpha, const Vector& x_true, double rho) {
    Vector diff(x_alpha.size());
    for (size_t i = 0; i < x_alpha.size(); ++i) {
        diff[i] = x_alpha[i] - x_true[i];
    }
    
    double norm_true = norm2(x_true);
    double eps_rel = (norm_true > 1e-300) ? norm2(diff) / norm_true : norm2(diff);
    double eta = norm2(x_alpha);
    
    return {eps_rel, rho, eta};
}