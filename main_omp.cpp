#include "./include/data_loader.h"
#include "./include/stats.h"
#include "./include/montecarlo.h"

#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <limits>

#include "./include/portfolio.h"

using namespace std;

int main()
{
    auto start = chrono::high_resolution_clock::now();
    const string priceFile = "../data/prices.csv";
    const string statsFile = "../data/stats.csv";
    const string covFile = "../data/covariance.csv";
    
    // 1) Build stats + covariance from prices
    computeStats(priceFile, statsFile, covFile);
    
    // 2) Load mean + covariance
    vector<string> symbols;
    vector<double> mean;
    vector<vector<double>> cov;
    loadStatsAndCovariance(statsFile, covFile, symbols, mean, cov);
    
    // 3) Run Parallel Monte Carlo across bias levels 0..5
    //int iterations = 10000000; //10 million iterations
    int iterations = 1000000; //1 million iterations
    int bias_levels = 6;
    int iterations_per_bias = iterations / bias_levels;

    double bestSharpe = -numeric_limits<double>::infinity();
    vector<double> bestWeights(mean.size(), 0.0);
    int winningBias = 0;

    for (int bias = 0; bias < bias_levels; bias++)
    {
        vector<double> candidateWeights(mean.size(), 0.0);
        double candidateSharpe = runMonteCarloLocal(
            iterations_per_bias,
            mean,
            cov,
            candidateWeights,
            bias);

        if (candidateSharpe > bestSharpe)
        {
            bestSharpe = candidateSharpe;
            bestWeights = candidateWeights;
            winningBias = bias;
        }
    }

    double bestReturn = portfolioReturn(bestWeights, mean);
    double bestRisk = portfolioRisk(bestWeights, cov);

    cout << "------------------------------------------------------------\n";
    cout << "PARALLEL RUN  | Threads = " << omp_get_max_threads() << " threads\n";
    cout << "------------------------------------------------------------\n";
    cout << fixed << setprecision(6);
    cout << left << setw(25) << "Best Sharpe:" << setw(12) << bestSharpe << "\n";
    cout << left << setw(25) << "Best Return:" << setw(12) << bestReturn << "\n";
    cout << left << setw(25) << "Best Risk:" << setw(12) << bestRisk << "\n";
    cout << left << setw(25) << "Winning bias_power:" << setw(12) << winningBias << "\n";

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    
    cout << left << setw(25) << "Time (sec):" << setw(12) << duration.count() << "\n";

    return 0;
}