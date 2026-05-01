#include <string>
#include <vector>

using namespace std;

void computeStats(
    const string& priceFile,
    const string& statsFile,
    const string& covFile);

double checkCovarianceStability(
    const vector<vector<double>>& cov_input,
    int N,
    int J,
    unsigned int seed = 42);