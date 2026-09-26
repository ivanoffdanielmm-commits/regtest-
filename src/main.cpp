#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix_ops.h"
#include "normal_eq.h"
#include "svd_jacobi.h"
#include "tikhonov.h"
#include "discrepancy.h"
#include "lcurve.h"
#include "metrics.h"

static Matrix identityMatrix(int n)
{
    Matrix I(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i)
        I[i][i] = 1.0;
    return I;
}

static Matrix hilbertMatrix(int n)
{
    Matrix H(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            H[i][j] = 1.0 / static_cast<double>(i + j + 1);
    return H;
}

static Matrix randomMatrix(int m, int n, std::mt19937 &gen)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix R(m, Vector(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            R[i][j] = dist(gen);
    return R;
}

static Vector matVec(const Matrix &A, const Vector &x)
{
    int m = A.size();
    int n = A[0].size();
    Vector y(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

static void runTest(const std::string &name,
                    const Matrix &A,
                    const Vector &x_true,
                    double noise_level,
                    unsigned seed)
{
    std::cout << "Тест: " << name << "\n";
    std::cout << "  m = " << A.size()
              << ", n = " << A[0].size()
              << ", noise = " << noise_level << "\n";

    int m = A.size();

    Vector y_exact = matVec(A, x_true);

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, noise_level);
    Vector y = y_exact;
    Vector noise(m, 0.0);
    for (int i = 0; i < m; ++i)
    {
        noise[i] = dist(gen);
        y[i] += noise[i];
    }
    double eps_noise = norm2(noise);

    Vector p(m, 1.0);
     Matrix A_tilde;
    Vector y_tilde;
    buildNormalEquations(A, y, p, A_tilde, y_tilde);

    Matrix U, V;
    Vector sigma;
    try
    {
        svdJacobi(A_tilde, U, V, sigma, 1e-10, 200);
        std::cout << "  [OK] SVD converged. sigma_min = "
                  << sigma.back() << ", sigma_max = " << sigma.front() << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "  [FAIL] SVD: " << e.what() << "\n";
        return;
    }

    try
    {
        Vector x_disc;
        double tau = 1.1;
        double alpha_disc = solveDiscrepancy(A_tilde, y_tilde, U, V, sigma,
                                             eps_noise, tau,
                                             1e-12, 1e1,
                                             1e-6, 60,
                                             &x_disc);

        double rho_disc = computeResidual(A_tilde, x_disc, y_tilde);
        Metrics m_disc = computeMetrics(x_disc, x_true, rho_disc);

        std::cout << "  [Discrepancy] alpha* = " << std::scientific
                  << std::setprecision(4) << alpha_disc
                  << ", eps_rel = " << m_disc.eps_rel
                  << ", rho = " << m_disc.rho
                  << ", tau*eps = " << tau * eps_noise << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "  [FAIL] Discrepancy: " << e.what() << "\n";
    }


}

int main()
{
    std::cout << std::scientific << std::setprecision(6);

    {
        int n = 5;
        Matrix A = identityMatrix(n);
        Vector x_true(n, 1.0);
        runTest("Identity 5x5", A, x_true, 0.01, 42);
    }

    {
        int n = 4;
        Matrix A = hilbertMatrix(n);
        Vector x_true(n, 1.0);
        runTest("Hilbert 4x4", A, x_true, 1e-3, 43);
    }

    {
        std::mt19937 gen(44);
        Matrix A = randomMatrix(6, 3, gen);
        Vector x_true = {1.0, -0.5, 2.0};
        runTest("Random 6x3", A, x_true, 0.05, 44);
    }

    return 0;
}
