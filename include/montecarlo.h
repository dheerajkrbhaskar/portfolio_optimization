#include <random>
#include <vector>

using namespace std;

void generateWeights(double* w, int N, int bias_power, std::mt19937& rng);

void runMonteCarlo(int iterations, const vector<double>& mean, const vector<std::vector<double>>& covariance);

double runMonteCarloLocal(
    int N,
    const vector<double>& mean,
    const vector<std::vector<double>>& cov,
    vector<double>& bestW,
    int bias_power
);