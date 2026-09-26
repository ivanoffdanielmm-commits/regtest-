#pragma once
#include "matrix_ops.h"
#include <string>
#include <vector>


struct Point {
    std::string name;   
    double      h0;     
};

struct Observation {
    std::string from;    
    std::string to;      
    double      observed;
    double      weight;  
};



std::vector<Point>       readPoints      (const std::string& filename);
std::vector<Observation> readObservations(const std::string& filename);


void buildLevelingSystem(const std::vector<Point>&       points,
                         const std::vector<Observation>& observations,
                         Matrix& A, Vector& y, Vector& p, Vector& x0);


void writeVectorCSV(const std::string& filename,
                    const Vector&      v,
                    const std::string& header = "");

void writeNamedVectorCSV(const std::string&              filename,
                         const std::vector<std::string>& names,
                         const Vector&                   v);

void writeMatrixCSV(const std::string& filename,
                    const Matrix&      M);

void writeLCurveData(const std::string& filename,
                     const Vector&      alphas,
                     const Vector&      rhos,
                     const Vector&      etas);


void writeAlphaErrors(const std::string& filename,
                      const Vector&      alphas,
                      const Vector&      rho,
                      const Vector&      eta);