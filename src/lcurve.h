#pragma once
#include "tikhonov.h"
#include <cmath>
#include <algorithm>
#include <cstddef>
static double smooth3(const Vector& v, int k)
{
    if (k == 0 || k == static_cast<int>(v.size()) - 1)
        return v[k];
    return (v[k - 1] + v[k] + v[k + 1]) / 3.0;
}

double findLCurveCorner(const Matrix& A_tilde,
                        const Vector& y_tilde,
                        const Matrix& U,
                        const Matrix& V,
                        const Vector& sigma,
                        const Vector& alphas,
                        Vector&       x_opt)
{
    const int n_alpha = static_cast<int>(alphas.size());
    if (n_alpha < 5)
    {
        throw std::invalid_argument(
            "findLCurveCorner: нужно минимум 5 точек для устойчивой оценки кривизны");
    }

    Vector eta_log(n_alpha), rho_log(n_alpha);
    const double eps_safe = 1e-300;


    for (int k = 0; k < n_alpha; ++k)
    {
        Vector x_alpha = tikhonovSolve(U, V, sigma, y_tilde, alphas[k]);
        double rho = computeResidual(A_tilde, x_alpha, y_tilde);
        double eta = norm2(x_alpha);

        eta_log[k] = std::log(std::max(eta, eps_safe));
        rho_log[k] = std::log(std::max(rho, eps_safe));
    }

    Vector eta_s(n_alpha), rho_s(n_alpha);
    for (int k = 0; k < n_alpha; ++k)
    {
        eta_s[k] = smooth3(eta_log, k);
        rho_s[k] = smooth3(rho_log, k);
    }

    double max_kappa = -1.0;
    int    k_star    = n_alpha / 2;

    for (int k = 1; k < n_alpha - 1; ++k)
    {
        double d_eta  = eta_s[k + 1] - eta_s[k - 1];
        double d_rho  = rho_s[k + 1] - rho_s[k - 1];
        double dd_eta = eta_s[k + 1] - 2.0 * eta_s[k] + eta_s[k - 1];
        double dd_rho = rho_s[k + 1] - 2.0 * rho_s[k] + rho_s[k - 1];

        double denom = d_eta * d_eta + d_rho * d_rho;
        if (denom < 1e-30)
            continue;

        double kappa = std::abs(d_eta * dd_rho - d_rho * dd_eta)
        / std::pow(denom, 1.5);

        if (kappa > max_kappa)
        {
            max_kappa = kappa;
            k_star    = k;
        }
    }

    x_opt = tikhonovSolve(U, V, sigma, y_tilde, alphas[k_star]);
    return alphas[k_star];
}
