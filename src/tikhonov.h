#pragma once
#include "matrix_ops.h"

Vector tikhonovSolve(const Matrix& U, const Matrix& V, const Vector& sigma, const Vector& y_tilde, double alpha) {
    int m = U.size();
    int n = U[0].size();
    Vector x_alpha(n, 0.0);

    for (int j = 0; j < n; ++j) {
        double c_j = 0.0;
        for (int i = 0; i < m; ++i) {
            c_j += U[i][j] * y_tilde[i];
        }
        
        double denom = sigma[j] * sigma[j] + alpha;
        double f_j = (denom > 0) ? sigma[j] / denom : 0.0;
        
        for (int k = 0; k < n; ++k) {
            x_alpha[k] += f_j * c_j * V[k][j];
        }
    }
    return x_alpha;
}

double computeResidual(const Matrix& A_tilde, const Vector& x_alpha, const Vector& y_tilde) {
    int m = A_tilde.size();
    int n = A_tilde[0].size();
    Vector r(m, 0.0);
    
    for (int i = 0; i < m; ++i) {
        double ax = 0.0;
        for (int j = 0; j < n; ++j) {
            ax += A_tilde[i][j] * x_alpha[j];
        }
        r[i] = ax - y_tilde[i];
    }
    return norm2(r);
}