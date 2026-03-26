#include "../include/data_loader.h"

#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;


// ================================
// Read symbols from prices.csv
// ================================

vector<string> readSymbolsFromPrices(const string& priceFile)
{
    ifstream file(priceFile);

    vector<string> symbols;

    if (!file.is_open())
    {
        cerr << "Error opening " << priceFile << endl;
        return symbols;
    }

    string line;
    getline(file, line); // header

    stringstream ss(line);

    string col;

    getline(ss, col, ','); // skip Date

    while (getline(ss, col, ','))
        symbols.push_back(col);

    file.close();

    return symbols;
}


// ================================
// Read prices matrix
// ================================

vector<vector<double>> readPrices(const string& priceFile)
{
    ifstream file(priceFile);

    vector<vector<double>> prices;

    if (!file.is_open())
    {
        cerr << "Error opening " << priceFile << endl;
        return prices;
    }

    string line;

    // skip header
    getline(file, line);

    while (getline(file, line))
    {
        stringstream ss(line);

        string value;

        vector<double> row;

        getline(ss, value, ','); // skip Date

        while (getline(ss, value, ','))
        {
            if (!value.empty())
                row.push_back(stod(value));
        }

        if (!row.empty())
            prices.push_back(row);
    }

    file.close();

    return prices;
}


// ======================================
// Load stats.csv + covariance.csv
// ======================================

void loadStatsAndCovariance(
    const string& statsFile,
    const string& covFile,
    vector<string>& symbols,
    vector<double>& mean,
    vector<vector<double>>& cov
)
{
    symbols.clear();
    mean.clear();
    cov.clear();


    // =====================
    // Read stats.csv
    // =====================

    ifstream fmean(statsFile);

    if (!fmean.is_open())
    {
        cerr << "Error opening " << statsFile << endl;
        return;
    }

    string line;

    getline(fmean, line); // header

    while (getline(fmean, line))
    {
        stringstream ss(line);

        string name, m, risk;

        getline(ss, name, ',');
        getline(ss, m, ',');
        getline(ss, risk, ',');

        symbols.push_back(name);
        mean.push_back(stod(m));
    }

    fmean.close();


    // =====================
    // Read covariance.csv
    // =====================

    ifstream fcov(covFile);

    if (!fcov.is_open())
    {
        cerr << "Error opening " << covFile << endl;
        return;
    }

    getline(fcov, line); // header row

    while (getline(fcov, line))
    {
        stringstream ss(line);

        vector<double> row;

        string val;

        while (getline(ss, val, ','))
        {
            if (!val.empty())
                row.push_back(stod(val));
        }

        if (!row.empty())
            cov.push_back(row);
    }

    fcov.close();
}