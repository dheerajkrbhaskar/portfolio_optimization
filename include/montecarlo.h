#include<vector>
using namespace std;

void runMonteCarlo(int iterations,const vector<double>& mean,const vector<std::vector<double>>& covariance);

double runMonteCarloLocal(
    int N,
    const vector<double>& mean,
    const vector<std::vector<double>>& cov,
    vector<double>& bestW
);