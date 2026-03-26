#include "../include/montecarlo.h"
#include "../include/portfolio.h"

#include <iostream>
#include <random>
#include <iomanip>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// For Sequential and OpenMP versions
void runMonteCarlo(int N, const vector<double> &mean, const vector<vector<double>> &covariance)
{
    const int m = mean.size();

    double bestSharpe = -1e18;
    vector<double> bestWeights;

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        mt19937 rng(1234 + tid); // different seed per thread

        double localBestSharpe = -1e18;
        vector<double> localBestWeights(m);

#pragma omp for schedule(guided)
        for (int k = 0; k < N; ++k)
        {
            vector<double> w = randomWeights(m, rng);

            const double ret = portfolioReturn(w, mean);
            const double risk = portfolioRisk(w, covariance);

            if (risk <= 0.0)
                continue;

            const double sharpe = ret / risk;

            if (sharpe > localBestSharpe)
            {
                localBestSharpe = sharpe;
                localBestWeights = w;
            }
        }
#pragma omp critical
        {
            if (localBestSharpe > bestSharpe)
            {
                bestSharpe = localBestSharpe;
                bestWeights = localBestWeights;
            }
        }
    }
    double bestReturn = portfolioReturn(bestWeights, mean);
    double bestRisk = portfolioRisk(bestWeights, covariance);

    cout << "------------------------------------------------------------\n";
#ifdef _OPENMP
    cout << "PARALLEL RUN  | Threads = " << omp_get_max_threads() << " threads\n";
#else
    cout << "SEQUENTIAL RUN\n";
#endif
    cout << "------------------------------------------------------------\n";
    cout << fixed << setprecision(6);
    cout << left << setw(25) << "Best Sharpe:" << setw(12) << bestSharpe << "\n";

    cout << left << setw(25) << "Best Return:" << setw(12) << bestReturn << "\n";

    cout << left << setw(25) << "Best Risk:" << setw(12) << bestRisk << "\n";
}

// For MPI version, returns local best Sharpe ratio(used in main_mpi.cpp)
double runMonteCarloLocal(
    int N,
    const vector<double> &mean,
    const vector<vector<double>> &cov,
    vector<double> &bestW)
{
#ifdef _OPENMP
    cout << "PARALLEL RUN  | Threads = " << omp_get_max_threads() << " threads\n";
#else
    cout << "SEQUENTIAL RUN\n";
#endif
    int m = mean.size();

    double bestSharpe = -1e18;
    bestW.assign(m, 0.0);

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        mt19937 rng(1234 + tid);

        double localBest = -1e18;
        vector<double> localBestW(m);

#pragma omp for schedule(guided)
        for (int k = 0; k < N; k++)
        {
            vector<double> w = randomWeights(m, rng);

            double ret = portfolioReturn(w, mean);
            double risk = portfolioRisk(w, cov);

            if (risk <= 0)
                continue;

            double s = ret / risk;

            if (s > localBest)
            {
                localBest = s;
                localBestW = w;
            }
        }

#pragma omp critical
        {
            if (localBest > bestSharpe)
            {
                bestSharpe = localBest;
                bestW = localBestW;
            }
        }
    }

    return bestSharpe;
}