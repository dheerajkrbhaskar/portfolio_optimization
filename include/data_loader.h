#include <string>
#include <vector>

using namespace std;

// Reads symbols from first row of prices.csv (skips first Date column)
vector<string> readSymbolsFromPrices(const string& priceFile);

// Reads numeric price matrix from prices.csv (rows=time, cols=assets)
vector<vector<double>> readPrices(const string& priceFile);

// Loads symbols + mean from stats.csv and covariance from covariance.csv
void loadStatsAndCovariance(
    const string& statsFile,
    const string& covFile,
    vector<string>& symbols,
    vector<double>& mean,
    vector<vector<double>>& cov
);