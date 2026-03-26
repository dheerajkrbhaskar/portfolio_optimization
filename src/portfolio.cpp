#include "../include/portfolio.h"
#include <cmath>
#include <random>

using namespace std;

vector<double> randomWeights(int n, mt19937 &rng)
{
    vector<double> w(n, 0.0);
    uniform_real_distribution<double> dist(0.0, 1.0);

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        w[i] = dist(rng);
        sum += w[i];
    }

    for (int i = 0; i < n; i++)
    {
        w[i] /= sum;
    }

    return w;
}

double portfolioReturn(const vector<double> &w, const vector<double> &mean)
{
    double r = 0.0;
    for (int i = 0; i < static_cast<int>(w.size()); i++)
    {
        r += w[i] * mean[i];
    }
    return r;
}

double portfolioRisk(const vector<double> &w, const vector<vector<double>> &cov)
{
    int n = static_cast<int>(w.size());
    double var = 0.0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            var += w[i] * w[j] * cov[i][j];
        }
    }

    return sqrt(var);
}