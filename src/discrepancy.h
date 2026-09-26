#pragma once
#include "matrix_ops.h"
#include "tikhonov.h"
#include <stdexcept>
inline double solveDiscrepancy(const Matrix &A_tilde, const Vector &y_tilde,
                               const Matrix &U, const Matrix &V, const Vector &sigma,
                               double eps, double tau,
                               double alpha_min = 1e-12,
                               double alpha_max = 1e1,
                               double delta_alpha = 1e-6,
                               int max_expansions = 60,
                               Vector *x_opt_out = nullptr)
{
    if (eps <= 0.0 || tau <= 1.0)
        throw std::invalid_argument(
            "solveDiscrepancy: требуется eps > 0 и tau > 1");

    auto d_fun = [&](double alpha) -> double
    {
        Vector x = tikhonovSolve(U, V, sigma, y_tilde, alpha);
        double rho = computeResidual(A_tilde, x, y_tilde);
        return rho - tau * eps;
    };

    double d_min = d_fun(alpha_min);
    double d_max = d_fun(alpha_max);

    int exp_left = 0;
    while (d_min > 0.0 && exp_left < max_expansions)
    {
        alpha_min /= 10.0;
        d_min = d_fun(alpha_min);
        ++exp_left;
    }

    int exp_right = 0;
    while (d_max < 0.0 && exp_right < max_expansions)
    {
        alpha_max *= 10.0;
        d_max = d_fun(alpha_max);
        ++exp_right;
    }

    if (d_min * d_max > 0.0)
        throw std::runtime_error(
            "solveDiscrepancy: не удалось зажать корень. "
            "Проверьте уровень шума ε и параметр τ.");

    double alpha_L = alpha_min;
    double alpha_R = alpha_max;
    double f_L = d_min;
    double alpha_M = 0.5 * (alpha_L + alpha_R);

    while ((alpha_R - alpha_L) > delta_alpha)
    {
        alpha_M = 0.5 * (alpha_L + alpha_R);
        double f_M = d_fun(alpha_M);

        if (f_M == 0.0)
            break;

        if (f_L * f_M < 0.0)
        {
            alpha_R = alpha_M;
        }
        else
        {
            alpha_L = alpha_M;
            f_L = f_M;
        }
    }

    alpha_M = 0.5 * (alpha_L + alpha_R);

    if (x_opt_out)
        *x_opt_out = tikhonovSolve(U, V, sigma, y_tilde, alpha_M);

    return alpha_M;
}
