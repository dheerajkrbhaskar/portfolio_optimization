#include <mpi.h>

#include "montecarlo.h"
#include "data_loader.h"
#include "portfolio.h"
#include "stats.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
    auto start = chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<string> symbols;
    vector<double> mean;
    vector<vector<double>> cov;

    int m = 0;

    if (rank == 0)
    {
        loadStatsAndCovariance(
            "../data/stats.csv",
            "../data/covariance.csv",
            symbols,
            mean,
            cov);

        m = mean.size();

        double delta_1k = checkCovarianceStability(cov, m, 1000);
        double delta_10k = checkCovarianceStability(cov, m, 10000);
        double delta_100k = checkCovarianceStability(cov, m, 100000);

        cout << "Covariance stability check:\n";
        cout << "  J=1,000   delta=" << delta_1k << (delta_1k > 0.1 ? "  [UNSTABLE]" : "  [ok]") << "\n";
        cout << "  J=10,000  delta=" << delta_10k << (delta_10k > 0.02 ? "  [marginal]" : "  [ok]") << "\n";
        cout << "  J=100,000 delta=" << delta_100k << (delta_100k > 0.02 ? "  [marginal]" : "  [ok]") << "\n";
    }

    // broadcast m to all processes
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // resize mean and cov on all processes
    mean.resize(m);
    cov.resize(m, vector<double>(m));

    // broadcast mean to all processes
    MPI_Bcast(mean.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // broadcast covariance to all processes
    for (int i = 0; i < m; i++)
    {
        MPI_Bcast(cov[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int N = 10000000; //10 million iterations across all processes
    int localN = N / size; //Iterations per process
    int bias_power = rank % 6;

    vector<double> localW(m);
    double localBest = runMonteCarloLocal(localN, mean, cov, localW, bias_power);

    //gather sharpe ratios to root process
    double globalBestSharpe;
    vector<double> allSharpe( size);
    MPI_Gather(&localBest, 1, MPI_DOUBLE, allSharpe.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //gather weights to root process
    vector<double> allLocalW(m);
    if(rank==0) allLocalW.resize(m * size);

    MPI_Gather(localW.data(), m, MPI_DOUBLE, allLocalW.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    if (rank == 0)
    {
        int bestIdx = 0;
        for (int i = 1; i < size; i++)
        {
            if (allSharpe[i] > allSharpe[bestIdx])
            {
                bestIdx = i;
            }
        }
        vector<double> bestW(m);
        for(int j=0; j<m; j++)
        {
            bestW[j] = allLocalW[bestIdx * m + j];
        }
        double globalBestSharpe = allSharpe[bestIdx];
        
        double bestReturn = portfolioReturn(bestW, mean);
        double bestRisk = portfolioRisk(bestW, cov);

        cout << "------------------------------------------------------------\n";
        cout << "MPI+OMP RUN  | Processes = " << size << " processes\n";
        cout << "------------------------------------------------------------\n";
        cout << fixed << setprecision(6);
        cout << left << setw(25) << "Best Sharpe:" << setw(12) << globalBestSharpe << "\n";
        cout << left << setw(25) << "Best Return:" << setw(12) << bestReturn << "\n";
        cout << left << setw(25) << "Best Risk:" << setw(12) << bestRisk << "\n";
        cout << left << setw(25) << "Winning bias_power:" << setw(12) << (bestIdx % 6) << "\n";
    }

    MPI_Finalize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << left << setw(25) << "Time (sec):" << setw(12) << duration.count() << "\n";
    return 0;
}