#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <map>
#include <cstdlib>
#include <cmath>

static std::string trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitCSV(const std::string &line, char delim = ',')
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream ss(line);
    while (std::getline(ss, token, delim))
        tokens.push_back(trim(token));
    return tokens;
}

static bool parseDouble(const std::string &s, double &out)
{
    if (s.empty())
        return false;
    char *end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return (end != s.c_str() && *end == '\0');
}

std::vector<Point> readPoints(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    std::vector<Point> points;
    std::string line;
    while (std::getline(file, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        auto tokens = splitCSV(line);
        if (tokens.size() < 2)
            continue;

        double h0;
        if (!parseDouble(tokens[1], h0))
            continue;

        Point p;
        p.name = tokens[0];
        p.h0 = h0;
        points.push_back(p);
    }
    return points;
}

std::vector<Observation> readObservations(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    std::vector<Observation> obs;
    std::string line;
    while (std::getline(file, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        auto tokens = splitCSV(line);
        if (tokens.size() < 4)
            continue;

        double observed, weight;
        if (!parseDouble(tokens[2], observed))
            continue;
        if (!parseDouble(tokens[3], weight))
            continue;

        Observation o;
        o.from = tokens[0];
        o.to = tokens[1];
        o.observed = observed;
        o.weight = weight;
        obs.push_back(o);
    }
    return obs;
}

void buildLevelingSystem(const std::vector<Point> &points,
                         const std::vector<Observation> &observations,
                         Matrix &A, Vector &y, Vector &p, Vector &x0)
{
    int n = static_cast<int>(points.size());
    int m = static_cast<int>(observations.size());

    if (n == 0 || m == 0)
        throw std::runtime_error("пустая сеть или пустые наблюдения");

    std::map<std::string, int> idx;
    x0.assign(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        idx[points[i].name] = i;
        x0[i] = points[i].h0;
    }

    A.assign(m, Vector(n, 0.0));
    y.assign(m, 0.0);
    p.assign(m, 0.0);

    for (int i = 0; i < m; ++i)
    {
        const Observation &o = observations[i];

        auto it_from = idx.find(o.from);
        auto it_to = idx.find(o.to);
        if (it_from == idx.end())
            throw std::runtime_error("неизвестный пункт " + o.from);
        if (it_to == idx.end())
            throw std::runtime_error("неизвестный пункт " + o.to);

        int i_from = it_from->second;
        int i_to = it_to->second;

        A[i][i_from] = -1.0;
        A[i][i_to] = +1.0;

        double h_from = points[i_from].h0;
        double h_to = points[i_to].h0;

        y[i] = o.observed - (h_to - h_from);
        p[i] = o.weight;
    }
}

void writeVectorCSV(const std::string &filename,
                    const Vector &v,
                    const std::string &header)
{
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    out << std::scientific << std::setprecision(10);

    if (!header.empty())
        out << header << "\n";

    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i + 1 < v.size())
            out << ",";
    }
    out << "\n";
}

void writeNamedVectorCSV(const std::string &filename,
                         const std::vector<std::string> &names,
                         const Vector &v)
{
    if (names.size() != v.size())
        throw std::invalid_argument("размерности не совпадают");

    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    out << std::scientific << std::setprecision(10);
    out << "name,value\n";
    for (size_t i = 0; i < v.size(); ++i)
        out << names[i] << "," << v[i] << "\n";
}

void writeMatrixCSV(const std::string &filename, const Matrix &M)
{
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    out << std::scientific << std::setprecision(10);
    for (const auto &row : M)
    {
        for (size_t j = 0; j < row.size(); ++j)
        {
            out << row[j];
            if (j + 1 < row.size())
                out << ",";
        }
        out << "\n";
    }
}

static void writeAlphaRhoEtaImpl(const std::string& filename,
                                 const Vector&      alphas,
                                 const Vector&      rho,
                                 const Vector&      eta)
{
    if (alphas.size() != rho.size() || alphas.size() != eta.size())
        throw std::invalid_argument("размерности не совпадают");

    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("не удалось открыть " + filename);

    out << std::scientific << std::setprecision(10);
    out << "alpha,rho,eta\n";
    for (size_t k = 0; k < alphas.size(); ++k)
        out << alphas[k] << "," << rho[k] << "," << eta[k] << "\n";
}

void writeLCurveData(const std::string& filename,
                     const Vector&      alphas,
                     const Vector&      rhos,
                     const Vector&      etas)
{
    writeAlphaRhoEtaImpl(filename, alphas, rhos, etas);
}

void writeAlphaErrors(const std::string& filename,
                      const Vector&      alphas,
                      const Vector&      rho,
                      const Vector&      eta)
{
    writeAlphaRhoEtaImpl(filename, alphas, rho, eta);
}