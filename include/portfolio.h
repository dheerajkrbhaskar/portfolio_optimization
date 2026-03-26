#include<vector>
#include<random>

using namespace std;

vector<double> randomWeights(int n, mt19937 &rng);

double portfolioReturn(
    const vector<double>& w,
    const vector<double>& mean);

double portfolioRisk(
    const vector<double>& w,
    const vector<vector<double>>& cov);