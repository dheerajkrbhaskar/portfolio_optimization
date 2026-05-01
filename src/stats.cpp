#include "stats.h"
#include "data_loader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

static bool choleskyDecompose(const vector<vector<double>>& A, vector<vector<double>>& L)
{
    int n = static_cast<int>(A.size());
    L.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double s = 0.0;
            for (int k = 0; k < j; k++)
            {
                s += L[i][k] * L[j][k];
            }

            if (i == j)
            {
                double diag = A[i][i] - s;
                if (diag <= 0.0)
                {
                    return false;
                }
                L[i][j] = sqrt(diag);
            }
            else
            {
                if (L[j][j] <= 0.0)
                {
                    return false;
                }
                L[i][j] = (A[i][j] - s) / L[j][j];
            }
        }
    }

    return true;
}

void computeStats(const string &priceFile, const string &statsFile, const string &covFile)
{
// #ifdef _OPENMP
//     cout << "Parallel mean,covariance computations with " << omp_get_max_threads() << " threads\n";
// #else
//     cout << "Sequential mean,covariance computations\n";
// #endif

    vector<string> symbols = readSymbolsFromPrices(priceFile);
    vector<vector<double>> prices = readPrices(priceFile);

    if (prices.size() < 2)
        throw runtime_error("Need at least 2 price rows");

    if (symbols.empty())
        throw runtime_error("No symbols found");

    int n = prices.size();
    int m = prices[0].size();

    // =============================
    // LOG RETURNS
    // =============================

    vector<vector<double>> returns(n - 1, vector<double>(m, 0.0));

#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 1; t < n; t++)
    {
        for (int j = 0; j < m; j++)
        {
            double p1 = prices[t - 1][j];
            double p2 = prices[t][j];

            if (p1 <= 0.0 || p2 <= 0.0)
                continue; // do not throw inside omp

            returns[t - 1][j] = log(p2 / p1);
        }
    }

    int T = n - 1;

    // =============================
    // MEAN
    // =============================

    vector<double> mean(m, 0.0);

#pragma omp parallel for schedule(static)
    for (int j = 0; j < m; j++)
    {
        double s = 0.0;

        for (int t = 0; t < T; t++)
            s += returns[t][j];

        mean[j] = s / T;
    }

    // =============================
    // COVARIANCE
    // =============================

    vector<vector<double>> cov(
        m,
        vector<double>(m, 0.0));

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double s = 0.0;

            for (int t = 0; t < T; t++)
            {
                s += (returns[t][i] - mean[i]) * (returns[t][j] - mean[j]);
            }

            cov[i][j] = s / (T - 1);
        }
    }

    // =============================
    // RISK
    // =============================

    vector<double> risk(m, 0.0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        risk[i] = sqrt(cov[i][i]);
    }

    // =============================
    // WRITE stats.csv
    // =============================

    ofstream statsOut(statsFile);

    if (!statsOut.is_open())
        throw runtime_error("Cannot open stats file");

    statsOut << "Stock,Mean,Risk\n";

    for (int i = 0; i < m; i++)
    {
        statsOut
            << symbols[i] << ","
            << mean[i] << ","
            << risk[i] << "\n";
    }

    statsOut.close();

    // =============================
    // WRITE covariance.csv
    // =============================

    ofstream covOut(covFile);

    if (!covOut.is_open())
        throw runtime_error("Cannot open cov file");

    for (int j = 0; j < m; j++)
    {
        covOut << symbols[j];
        if (j < m - 1)
            covOut << ",";
    }

    covOut << "\n";

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            covOut << cov[i][j];

            if (j < m - 1)
                covOut << ",";
        }

        covOut << "\n";
    }

    covOut.close();

    // cout << "Mean, risk, and covariance computed\n";
    cout << "Saved " << statsFile << " and " << covFile << "\n";
}

double checkCovarianceStability(
    const vector<vector<double>>& cov_input,
    int N,
    int J,
    unsigned int seed)
{
    if (N <= 0 || J <= 1)
    {
        throw runtime_error("Invalid N or J in checkCovarianceStability");
    }

    vector<vector<double>> stabilized = cov_input;
    vector<vector<double>> L;

    bool ok = choleskyDecompose(stabilized, L);
    if (!ok)
    {
        double jitter = 1e-10;
        for (int attempt = 0; attempt < 8 && !ok; attempt++)
        {
            for (int i = 0; i < N; i++)
            {
                stabilized[i][i] += jitter;
            }
            ok = choleskyDecompose(stabilized, L);
            jitter *= 10.0;
        }
    }

    if (!ok)
    {
        throw runtime_error("Covariance matrix is not positive definite even after jitter");
    }

    mt19937 rng(seed);
    normal_distribution<double> ndist(0.0, 1.0);

    vector<vector<double>> samples(J, vector<double>(N, 0.0));
    for (int t = 0; t < J; t++)
    {
        vector<double> z(N, 0.0);
        for (int i = 0; i < N; i++)
        {
            z[i] = ndist(rng);
        }

        for (int i = 0; i < N; i++)
        {
            double v = 0.0;
            for (int k = 0; k <= i; k++)
            {
                v += L[i][k] * z[k];
            }
            samples[t][i] = v;
        }
    }

    vector<double> mean(N, 0.0);
    for (int i = 0; i < N; i++)
    {
        double s = 0.0;
        for (int t = 0; t < J; t++)
        {
            s += samples[t][i];
        }
        mean[i] = s / static_cast<double>(J);
    }

    vector<vector<double>> realized(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double s = 0.0;
            for (int t = 0; t < J; t++)
            {
                s += (samples[t][i] - mean[i]) * (samples[t][j] - mean[j]);
            }
            realized[i][j] = s / static_cast<double>(J - 1);
        }
    }

    double delta = 0.0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double d = fabs(realized[i][j] - cov_input[i][j]);
            delta = max(delta, d);
        }
    }

    return delta;
}