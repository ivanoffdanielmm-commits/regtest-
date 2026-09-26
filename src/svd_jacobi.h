#pragma once
#include "matrix_ops.h"
#include <omp.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <string>

void svdJacobi(const Matrix &A_tilde, Matrix &U, Matrix &V,
               Vector &sigma, double eps = 1e-9, int K_max = 100)
{
    int m = A_tilde.size();
    int n = A_tilde[0].size();

    if (m < n)
        throw std::invalid_argument("svdJacobi: требуется m >= n");

    U = A_tilde;
    V.assign(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i)
        V[i][i] = 1.0;
    sigma.assign(n, 0.0);

    int k = 0;
    bool converged = false;

    while (k < K_max && !converged)
    {
        converged = true;

        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;

#pragma omp parallel for reduction(+ : alpha, beta, gamma) if (m > 64)
                for (int i = 0; i < m; ++i)
                {
                    alpha += U[i][p] * U[i][p];
                    beta += U[i][q] * U[i][q];
                    gamma += U[i][p] * U[i][q];
                }

                if (alpha < 1e-300 || beta < 1e-300)
                    continue;

                if (std::abs(gamma) / std::sqrt(alpha * beta) > eps)
                {
                    converged = false;

                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = (zeta >= 0 ? 1.0 : -1.0) /
                               (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = c * t;

#pragma omp parallel for if (m > 64)
                    for (int i = 0; i < m; ++i)
                    {
                        double u_ip = U[i][p];
                        double u_iq = U[i][q];
                        U[i][p] = c * u_ip - s * u_iq;
                        U[i][q] = s * u_ip + c * u_iq;
                    }

#pragma omp parallel for if (n > 64)
                    for (int i = 0; i < n; ++i)
                    {
                        double v_ip = V[i][p];
                        double v_iq = V[i][q];
                        V[i][p] = c * v_ip - s * v_iq;
                        V[i][q] = s * v_ip + c * v_iq;
                    }
                }
            }
        }
        ++k;
    }

    if (!converged)
        throw std::runtime_error(
            "svdJacobi: алгоритм не сошёлся за K_max = " +
            std::to_string(K_max) + " итераций");

    for (int j = 0; j < n; ++j)
    {
        double norm_col = 0.0;
        for (int i = 0; i < m; ++i)
            norm_col += U[i][j] * U[i][j];
        sigma[j] = std::sqrt(norm_col);

        if (sigma[j] > 1e-300)
        {
            for (int i = 0; i < m; ++i)
                U[i][j] /= sigma[j];
        }
        else
        {
            for (int i = 0; i < m; ++i)
                U[i][j] = 0.0;
        }
    }

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b)
              { return sigma[a] > sigma[b]; });

    Matrix U_sorted(m, Vector(n));
    Matrix V_sorted(n, Vector(n));
    Vector sigma_sorted(n);

    for (int j = 0; j < n; ++j)
    {
        sigma_sorted[j] = sigma[idx[j]];
        for (int i = 0; i < m; ++i)
            U_sorted[i][j] = U[i][idx[j]];
        for (int i = 0; i < n; ++i)
            V_sorted[i][j] = V[i][idx[j]];
    }

    U = std::move(U_sorted);
    V = std::move(V_sorted);
    sigma = std::move(sigma_sorted);
}