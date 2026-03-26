#include "./include/data_loader.h"
#include "./include/stats.h"
#include "./include/montecarlo.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

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

    // 3) Run sequential Monte Carlo
    int iterations = 10000000; //10 million iterations
    // int iterations = 1000000; //1 million iterations

    runMonteCarlo(iterations, mean, cov);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << left << setw(25) << "Time (sec):" << setw(12) << duration.count() << "\n";

    return 0;
}