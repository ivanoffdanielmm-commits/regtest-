#pragma once
#include "matrix_ops.h"
#include <cmath>

void buildNormalEquations(const Matrix& A,
                          const Vector& y,
                          const Vector& p,
                          Matrix&       A_tilde,
                          Vector&       y_tilde)
{
    int m = A.size();
    int n = A[0].size();

    A_tilde.assign(m, Vector(n, 0.0));
    y_tilde.assign(m, 0.0);

    for (int i = 0; i < m; ++i)
    {
        double sqrt_p = std::sqrt(p[i]);
        y_tilde[i] = sqrt_p * y[i];
        for (int j = 0; j < n; ++j)
            A_tilde[i][j] = sqrt_p * A[i][j];
    }
}