# Parallel Portfolio Optimization using Monte Carlo Simulation (OpenMP & MPI)

---

## 1. Introduction

This project implements **Portfolio Optimization for Financial Risk Diversification** based on the **Markowitz Modern Portfolio Theory**, developed by **Harry Markowitz**.

The primary objective is to construct an optimal investment portfolio that:

* **Minimizes risk (variance)**
* For a given level of expected return

Mathematically, portfolio risk is defined as:

```math
\sigma^2 = w^T \Sigma w
```

Where:

* **w** → vector of asset weights
* **Σ (Sigma)** → covariance matrix of asset returns

Since this model involves **matrix operations and large-scale simulations**, it is highly suitable for **parallel computing techniques** such as:

* OpenMP (shared memory)
* MPI (distributed memory)

---

## 2. Project Objective

* Build a **correct sequential implementation** of portfolio optimization
* Identify **computational bottlenecks** in the sequential version
* Apply **parallelization techniques** to optimize performance
* Compare performance across different models:

  * Sequential
  * OpenMP
  * MPI
  * MPI + OpenMP

---

##  3. Methodology

###  Data Collection & Preprocessing

* Input: `prices.csv`
* Compute:

  * Log returns
  * Mean returns
  * Covariance matrix

Generated files:

* `stats.csv`
* `covariance.csv`

---

###  Sequential Implementation

* Monte Carlo simulation generates random portfolios
* Computes:

  * Expected return
  * Risk (standard deviation)
  * Sharpe ratio
* Finds optimal portfolio

---

###  Parallel Implementation

#### OpenMP (CPU Parallelism)

* Parallelizes Monte Carlo iterations
* Uses:

  * Thread-level parallelism
  * Work sharing (`#pragma omp for`)
  * Guided scheduling

#### MPI (Distributed Computing)

* Distributes Monte Carlo workload across processes
* Each process computes:

  * Local best Sharpe ratio + weights
* Root process:

  * Collects results
  * Selects global best portfolio

#### Hybrid (MPI + OpenMP)

* MPI across nodes
* OpenMP within each node
* Achieves maximum performance

---

##  Core Computations

### Log Return

```math
r_t = \ln(P_t / P_{t-1})
```

### Portfolio Return

```math
R = \sum w_i \mu_i
```

### Portfolio Risk

```math
\sigma = \sqrt{w^T \Sigma w}
```

### Sharpe Ratio

```math
S = \frac{R}{\sigma}
```

---

## 🏗️ Project Structure

```id="projstruct"
portfolio_optimization/
│
├── data/
│   ├── prices.csv
│   ├── stats.csv
│   └── covariance.csv
│
├── include/
├── src/
│
├── main_seq.cpp
├── main_omp.cpp
├── main_mpi.cpp
│
├── CMakeLists.txt
└── build/
```

---

## ▶️ How to Run

### Build

```bash
rm -rf build
mkdir build
cd build
cmake ..
make
```

---

### Sequential

```bash
./seq
```

---

### OpenMP

```bash
export OMP_NUM_THREADS=4
./omp
```

---

### MPI

```bash
mpirun -np 4 ./mpi
```

---

### Hybrid

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./mpi
```

---

##  4. Performance Evaluation

The system is evaluated based on:

* **Execution Time**
* **Speedup**
* **Efficiency**
* **Scalability**

### Observations

* OpenMP provides **moderate speedup** over sequential execution
* MPI provides **significant speedup** by distributing workload
* Hybrid model gives **best performance** by utilizing both models

---

##  Key Optimizations

* Parallel Monte Carlo loop (major bottleneck)
* Efficient scheduling (`guided`, `static`)
* Thread-local random number generation
* MPI communication:

  * Broadcast (data sharing)
  * Gather (results aggregation)
* Avoiding race conditions using local variables

---

##  Example Output

```id="exampleout"
MPI RUN | Processes = 4

Best Sharpe: 0.069590
Best Return: 0.000716
Best Risk:   0.010295

Time (sec): 2
```

---

##  5. Expected Results

* Moderate speedup with OpenMP compared to sequential
* Significant speedup with MPI
* Improved scalability with hybrid model

---

##  6. Conclusion

This project demonstrates how **High-Performance Computing (HPC)** can significantly enhance:

* Large-scale financial simulations
* Portfolio optimization
* Risk analysis

By combining **Modern Portfolio Theory with parallel computing**, we achieve:

* Faster computations
* Better scalability
* Efficient resource utilization

---

##  Tech Stack

* C++
* OpenMP
* MPI (OpenMPI)
* CMake
* Linux / WSL / macOS

---

##  Author

**Dheeraj Bhaskar**


---

## If you found this useful

Consider giving this project a ⭐
