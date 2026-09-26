#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>

#include "matrix_ops.h"
#include "io.h"
#include "normal_eq.h"
#include "svd_jacobi.h"
#include "tikhonov.h"
#include "discrepancy.h"
#include "lcurve.h"
#include "metrics.h"

int main(int argc, char **argv)
{
    std::string net_file = "data/network.csv";
    std::string obs_file = "data/measurements.csv";
    std::string res_dir = "results/";

    if (argc > 1)
        net_file = argv[1];
    if (argc > 2)
        obs_file = argv[2];
    if (argc > 3)
        res_dir = argv[3];
    if (!res_dir.empty() && res_dir.back() != '/')
        res_dir += '/';

    std::cout << std::scientific << std::setprecision(6);

    try
    {
        // 1. Чтение
        std::vector<Point> points = readPoints(net_file);
        std::vector<Observation> obs = readObservations(obs_file);

        std::cout << "Прочитано пунктов:    " << points.size() << "\n";
        std::cout << "Прочитано наблюдений: " << obs.size() << "\n";

        Matrix A;
        Vector y, p, x0;
        buildLevelingSystem(points, obs, A, y, p, x0);

        int m = static_cast<int>(A.size());
        int n = static_cast<int>(A[0].size());
        std::cout << "Размер системы: m = " << m
                  << ", n = " << n
                  << ", избыточность = " << (m - n) << "\n";

        // 2. Нормальные уравнения
        Matrix A_tilde;
        Vector y_tilde;
        buildNormalEquations(A, y, p, A_tilde, y_tilde);

        // 3. SVD
        Matrix U, V;
        Vector sigma;
        svdJacobi(A_tilde, U, V, sigma, 1e-10, 500);

        std::cout << "SVD сошёлся. sigma_max = " << sigma.front()
                  << ", sigma_min = " << sigma.back() << "\n";

        // 4. Оценка уровня шума
        Vector x_ls = tikhonovSolve(U, V, sigma, y_tilde, 1e-15);
        double rho_ls = computeResidual(A_tilde, x_ls, y_tilde);
        int dof = std::max(1, m - n);
        double sigma_0 = rho_ls / std::sqrt(static_cast<double>(dof));
        double eps_est = sigma_0 * std::sqrt(static_cast<double>(m));

        std::cout << "Оценка СКО единицы веса sigma_0 = " << sigma_0 << "\n";
        std::cout << "Оценка уровня шума eps          = " << eps_est << "\n";

        // 5. Принцип невязки
        Vector x_disc;
        double tau = 1.1;
        double alpha_disc = solveDiscrepancy(A_tilde, y_tilde, U, V, sigma,
                                             eps_est, tau,
                                             1e-14, 1e3, 1e-8, 60,
                                             &x_disc);

        double rho_disc = computeResidual(A_tilde, x_disc, y_tilde);
        std::cout << "Discrepancy: alpha* = " << alpha_disc
                  << ", rho = " << rho_disc
                  << ", tau*eps = " << tau * eps_est << "\n";

        // 6. L-кривая
        const int N_alpha = 200;
        Vector alphas(N_alpha);
        for (int k = 0; k < N_alpha; ++k)
        {
            double t = static_cast<double>(k) / (N_alpha - 1);
            alphas[k] = std::pow(10.0, -8.0 + t * 8.0);
        }

        Vector x_lcurve;
        double alpha_lc = findLCurveCorner(A_tilde, y_tilde, U, V, sigma,
                                           alphas, x_lcurve);

        double rho_lc = computeResidual(A_tilde, x_lcurve, y_tilde);
        std::cout << "L-curve:     alpha* = " << alpha_lc
                  << ", rho = " << rho_lc << "\n";

        // 7. Сбор метрик по всей сетке
        Vector rhos(N_alpha), etas(N_alpha);
        for (int k = 0; k < N_alpha; ++k)
        {
            Vector xk = tikhonovSolve(U, V, sigma, y_tilde, alphas[k]);
            rhos[k] = computeResidual(A_tilde, xk, y_tilde);
            etas[k] = norm2(xk);
        }

        std::vector<std::string> names;
        names.reserve(points.size());
        for (const auto &p : points)
            names.push_back(p.name);

        writeNamedVectorCSV(res_dir + "corrections_ls.csv", names, x_ls);
        writeNamedVectorCSV(res_dir + "corrections_discrepancy.csv", names, x_disc);
        writeNamedVectorCSV(res_dir + "corrections_lcurve.csv", names, x_lcurve);

        Vector h_disc(n);
        for (int i = 0; i < n; ++i)
            h_disc[i] = x0[i] + x_disc[i];
        writeNamedVectorCSV(res_dir + "adjusted_heights.csv", names, h_disc);

        writeLCurveData(res_dir + "lcurve.csv", alphas, rhos, etas);

        writeAlphaErrors(res_dir + "error_vs_alpha.csv",
                         alphas, rhos, etas);

        std::cout << "\nРезультаты записаны в " << res_dir << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
