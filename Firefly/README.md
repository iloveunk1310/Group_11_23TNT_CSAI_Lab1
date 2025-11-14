# AI Search and Optimization Project

A comprehensive, production-ready Python framework for comparing **Firefly Algorithm** (FA) with classical optimization methods on continuous and discrete benchmark problems.

## 🎯 Project Overview

This project implements and benchmarks multiple optimization algorithms with:

- ✅ **Full type hints** for all functions and classes
- ✅ **Comprehensive error handling** with actionable error messages
- ✅ **>80% test coverage** with edge case testing
- ✅ **Parallel execution** support for faster benchmarking
- ✅ **COCO/BBOB-standard analysis** with runtime-centric metrics
- ✅ **Academic-grade visualizations** following metaheuristic best practices
- ✅ **Reproducible results** with fixed seeds
- ✅ **Statistical analysis** with Wilcoxon, Friedman, and effect sizes

### Algorithms Implemented

#### Swarm Intelligence
- **Firefly Algorithm (FA)** - Bio-inspired optimization
  - Continuous optimization variant
  - Discrete Knapsack variant with repair/penalty strategies

#### Classical Baselines
- **Hill Climbing (HC)** - Greedy local search with restart
- **Simulated Annealing (SA)** - Probabilistic local search with temperature scheduling
- **Genetic Algorithm (GA)** - Evolutionary optimization with elitism

### Benchmark Problems

#### Continuous Functions
- **Rastrigin** - Highly multimodal with many local minima
  - Dimensions: d=10, 30, 50
  - Global optimum: f(0,...,0) = 0
  - Domain: [-5.12, 5.12]^d
  - Multi-tier success thresholds (Gold/Silver/Bronze)

#### Discrete Problems
- **0/1 Knapsack** - Maximize value within capacity constraint
  - Sizes: n=50, 100, 200 items
  - 4 instance types: uncorrelated, weakly correlated, strongly correlated, subset-sum
  - DP optimal solution available for n ≤ 100
  - Multi-tier gap thresholds (1%/5%/10%)

## 📁 Project Structure

```
Firefly/
├── src/
│   ├── core/                      # Base classes and utilities
│   │   ├── base_optimizer.py      # Abstract optimizer interface
│   │   ├── problem_base.py        # Abstract problem interface
│   │   └── utils.py               # Helper functions
│   │
│   ├── problems/
│   │   ├── continuous/            # Continuous benchmark functions
│   │   │   └── rastrigin.py       # Rastrigin function
│   │   └── discrete/              # Discrete optimization problems
│   │       └── knapsack.py        # 0/1 Knapsack with DP solver
│   │
│   ├── swarm/                     # Swarm intelligence algorithms
│   │   └── fa.py                  # Firefly Algorithm (continuous & discrete)
│   │
│   └── classical/                 # Classical baseline algorithms
│       ├── hill_climbing.py
│       ├── simulated_annealing.py
│       └── genetic_algorithm.py
│
├── benchmark/                     # Comprehensive benchmark suite
│   ├── config.py                  # Centralized configurations
│   ├── instance_generator.py      # Knapsack instance generation
│   ├── run_rastrigin.py           # Rastrigin benchmark runner
│   ├── run_knapsack.py            # Knapsack benchmark runner
│   ├── analyze_results.py         # COCO/BBOB-standard analysis
│   ├── visualize.py               # Academic-grade visualizations
│   ├── run_all.py                 # Master script with parallelization
│   ├── test_benchmarks.py         # Integration tests
│   └── results/                   # Auto-generated results
│       ├── rastrigin/
│       ├── knapsack/
│       ├── plots/                 # All visualizations
│       ├── summaries/             # CSV analysis outputs
│       └── logs/
│
├── test/                          # Unit tests (>80% coverage)
│   ├── test_problems.py
│   ├── test_firefly_algorithm.py
│   ├── test_classical_algorithms.py
│   ├── test_edge_cases.py
│   ├── test_parallel_execution.py
│   └── run_all_tests.py
│
├── demo.py                        # Quick demonstration
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NumPy, SciPy, Matplotlib, Pandas
- pytest (for testing)

### Installation

```bash
git clone https://github.com/1234quan1234/Firefly.git
cd Firefly

# Using conda (recommended)
conda env create -f environment.yml
conda activate firefly

# Or using pip
pip install -r requirements.txt
```

### Quick Start

#### Run Complete Benchmark Suite (Recommended)

```bash
# Quick mode with parallel execution
python benchmark/run_all.py --quick --jobs -1

# Full benchmark (30 runs per config)
python benchmark/run_all.py --full --jobs 4
```

This will:
- Run all Rastrigin configurations with multi-tier targets
- Run all Knapsack instances with gap-based analysis
- Generate COCO/BBOB-standard metrics (ERT, ECDF, profiles)
- Create comprehensive visualizations
- Perform statistical testing with effect sizes

#### Run Individual Benchmarks

**Rastrigin:**
```bash
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4
python benchmark/run_rastrigin.py --config multimodal_escape --jobs -1
python benchmark/run_rastrigin.py --config scalability --jobs 4
```

**Knapsack:**
```bash
python benchmark/run_knapsack.py --size 50 --jobs 4
python benchmark/run_knapsack.py --size 100 --jobs -1
python benchmark/run_knapsack.py --size 200 --jobs 4
```

#### Analysis and Visualization

```bash
# Generate all analysis artifacts
python benchmark/analyze_results.py --problem all

# Generate all plots
python benchmark/visualize.py
```

## 💡 Key Features

### 1. Multi-Tier Success Analysis

**Rastrigin:**
- **Gold** 🥇: |f(x)| ≤ threshold_gold (e.g., 1.0 for d=10)
- **Silver** 🥈: |f(x)| ≤ threshold_silver (e.g., 10.0)
- **Bronze** 🥉: |f(x)| ≤ threshold_bronze (e.g., 30.0)

**Knapsack:**
- **Gold** 🥇: Gap ≤ 1% from DP optimal
- **Silver** 🥈: Gap ≤ 5%
- **Bronze** 🥉: Gap ≤ 10%

Each tier tracks:
- Success rate
- Hitting time (evaluations to first success)
- ECDF of runtime-to-target

### 2. COCO/BBOB-Standard Metrics

Following the COCO (COmparing Continuous Optimizers) framework:

- **Fixed-Target ECDF**: Runtime distribution to reach target
- **Expected Running Time (ERT)**: Average evaluations to success (with censoring)
- **Performance Profiles** (Dolan-Moré): Robustness across problems
- **Data Profiles** (Moré-Wild): Budget-quality tradeoffs
- **Fixed-Budget Analysis**: Quality achieved at budget checkpoints

### 3. Enhanced Statistical Analysis

- **Wilcoxon signed-rank test** with Holm correction
- **Vargha-Delaney A12** effect size
- **Cliff's delta** effect size
- **Friedman test** for multiple algorithms
- **Nemenyi post-hoc** with critical difference diagrams
- **Copeland ranking** for pairwise dominance

### 4. Diversity and Stagnation Analysis (Rastrigin)

- Population diversity tracking (normalized by √D)
- Diversity evolution: Initial → Mid-point → Final
- Stagnation detection (longest non-improvement period)
- Critical for diagnosing premature convergence

### 5. Corrected Gap Computation (Knapsack)

**Important:** Gap computation now correctly handles maximization:
```python
# CORRECT (for maximization problems)
gap = 100.0 * (optimal - achieved) / optimal

# Gap = 0% means optimal solution found
# Gap = 5% means achieved is 95% of optimal
```

Previous versions had sign errors that inflated gaps.

## 📊 Benchmark Configurations

### Rastrigin

| Config              | Dim | Budget | Max Iter | Gold   | Silver | Bronze | Purpose             |
| ------------------- | --- | ------ | -------- | ------ | ------ | ------ | ------------------- |
| `quick_convergence` | 10  | 10,000 | 250      | 1.0    | 10.0   | 30.0   | Fast convergence    |
| `multimodal_escape` | 30  | 30,000 | 500      | 5.0    | 25.0   | 50.0   | Escape local minima |
| `scalability`       | 50  | 50,000 | 625      | 10.0   | 50.0   | 80.0   | High-dimensional    |

**Algorithm Parameters:**
- FA: n=40, α=0.3, β₀=1.0, γ=1.0
- SA: T₀=100, cooling=0.95
- HC: neighbors=20, restart=50
- GA: pop=40, crossover=0.8, mutation=0.1

### Knapsack

| Size | Types              | Seeds        | Budget | Max Iter (FA/GA) | DP Optimal? |
| ---- | ------------------ | ------------ | ------ | ---------------- | ----------- |
| 50   | All 4 types        | 42, 123, 999 | 10,000 | 166              | ✓           |
| 100  | All 4 types        | 42, 123, 999 | 15,000 | 250              | ✓           |
| 200  | Uncorr, Weak only  | 42, 123, 999 | 30,000 | 500              | ✗           |

**Gap Thresholds:**
- Gold: 1% (near-optimal)
- Silver: 5% (good for pure metaheuristics)
- Bronze: 10% (acceptable baseline)

**Algorithm Parameters:**
- FA: n=60, α_flip=0.2, max_flips=3, repair=greedy
- SA: T₀=1000, cooling=0.95
- HC: neighbors=20, restart=100
- GA: pop=60, crossover=0.8, mutation=1/n, elitism=0.1

## 📈 Output Artifacts

### JSON Results (Compressed)

- Rastrigin: `rastrigin_{config}_{algo}_{scenario}_{timestamp}.json.gz`
- Knapsack: `knapsack_n{size}_{type}_seed{seed}_{algo}_{scenario}_{timestamp}.json.gz`

**Key improvements:**
- Multi-tier success tracking
- Hitting time recording
- Status tracking (ok/timeout/nan/error)
- Budget utilization monitoring
- History stored for all runs (including failures)

### CSV Summaries

Generated by `analyze_results.py`:

**Rastrigin:**
- `rastrigin_summary.csv`: Basic statistics
- `rastrigin_fixed_target_ecdf.csv`: ECDF data
- `rastrigin_ert.csv`: Expected running times
- `rastrigin_fixed_budget.csv`: Fixed-budget checkpoints
- `rastrigin_performance_profiles.csv`: Performance profile data
- `rastrigin_data_profiles.csv`: Data profile data
- `rastrigin_diversity_summary.csv`: Diversity metrics
- `rastrigin_stagnation.csv`: Stagnation analysis
- `rastrigin_pairwise_stats.csv`: Statistical test results

**Knapsack:**
- `knapsack_summary_by_instance.csv`: Per-instance statistics
- `knapsack_summary_by_type.csv`: Aggregated by type
- `knapsack_fixed_target_ecdf.csv`: ECDF for gap thresholds
- `knapsack_ert.csv`: Expected running times
- `knapsack_fixed_budget.csv`: Fixed-budget performance
- `knapsack_performance_profiles.csv`: Dolan-Moré profiles
- `knapsack_data_profiles.csv`: Moré-Wild profiles
- `knapsack_pairwise_stats.csv`: Statistical comparisons

### Visualizations

Generated by `visualize.py` in `benchmark/results/plots/`:

**Rastrigin:**
- Fixed-target ECDF (per config, per tier)
- ERT bar charts with confidence intervals
- Fixed-budget convergence curves
- Performance profiles (Dolan-Moré)
- Data profiles (Moré-Wild)
- Diversity panel (4 metrics)
- Stagnation analysis (histogram + ECDF)
- Anytime AUC comparison
- Success breakdown (multi-tier)
- Cross-scenario comparison

**Knapsack:**
- Fixed-target ECDF (if DP available)
- ERT bar charts (per gap threshold)
- Fixed-budget analysis (gap or value)
- Performance profiles (Dolan-Moré)
- Data profiles (Moré-Wild)
- Pairwise win-loss heatmaps
- Copeland ranking
- Success breakdown (multi-tier gaps)

## 🧪 Testing

```bash
# Run all unit tests
python test/run_all_tests.py

# Run specific test modules
python -m unittest test.test_problems
python -m unittest test.test_firefly_algorithm

# Run benchmark integration tests
python benchmark/test_benchmarks.py
```

## 📚 Documentation

- **README.md** (this file): Project overview
- **QUICKSTART.md**: Quick start guide with examples
- **benchmark/README.md**: Detailed benchmark documentation
- **test/README.md**: Testing framework documentation

## 🔬 Scientific Rigor

This framework follows best practices from optimization literature:

1. **COCO/BBOB Standards**: Runtime-centric analysis with fixed-target and fixed-budget perspectives
2. **Multiple Independent Runs**: 30 runs per configuration for statistical significance
3. **Multiple Comparisons Correction**: Holm-Bonferroni for family-wise error control
4. **Effect Size Reporting**: A12 and Cliff's delta alongside p-values
5. **Reproducibility**: Fixed seeds for all randomness sources
6. **Budget Control**: Strict evaluation counting with utilization tracking
7. **Failure Tracking**: All runs logged, including failures with error details

## 📖 References

**COCO Framework:**
- Hansen et al. (2016). "COCO: A Platform for Comparing Continuous Optimizers"

**Performance Analysis:**
- Dolan & Moré (2002). "Benchmarking optimization software with performance profiles"
- Moré & Wild (2009). "Benchmarking derivative-free optimization algorithms"

**Statistical Testing:**
- Demšar (2006). "Statistical comparisons of classifiers over multiple data sets"
- García & Herrera (2008). "An extension on statistical comparisons of classifiers"

**Effect Sizes:**
- Vargha & Delaney (2000). "A critique and improvement of the CL common language effect size"
- Cliff (1993). "Dominance statistics: Ordinal analyses to answer ordinal questions"

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Contact

[Add contact information]

## 🙏 Acknowledgments

This project implements benchmarking standards from the COCO/BBOB community and follows best practices from computational intelligence literature.
1. **Uncorrelated**: Random values and weights
2. **Weakly Correlated**: values ≈ weights ± noise
3. **Strongly Correlated**: values = weights + 100
4. **Subset-Sum**: values = weights (hardest)

**Algorithm Parameters:**

* **FA**: n_fireflies=60, α_flip=0.2, max_flips=3, repair="greedy_remove"
* **SA**: T₀=1000, cooling=0.95
* **HC**: neighbors=20, restart=100
* **GA**: pop=60, crossover=0.8, mutation=1/n, elitism=0.1

## 📈 Output Format

All benchmark results are saved in JSON format for reproducibility.

### Rastrigin Results

**File naming:** `benchmark/results/rastrigin/rastrigin_{config}_{algo}_{scenario}_{timestamp}.json`

```json
{
  "metadata": {
    "problem": "rastrigin",
    "config_name": "quick_convergence",
    "algorithm": "FA",
    "scenario": "out_of_the_box",
    "timestamp": "20251110T200402",
    "dimension": 10,
    "budget": 10000,
    "max_iter": 250,
    "pop_size": 40,
    "problem_seed": 42,
    "n_runs": 30,
    "n_successful": 28,
    "n_failed": 2,
    "status_breakdown": {
      "ok": 28,
      "timeout": 1,
      "nan": 1
    },
    "thresholds_used": {
      "gold": 1.0,
      "silver": 5.0,
      "bronze": 10.0
    },
    "avg_budget_utilization": 0.998
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "problem_seed": 42,
      "best_fitness": 8.4567,
      "history": [45.6, 34.2, 23.1, 15.8, 8.4567],
      "elapsed_time": 2.15,
      "evaluations": 10000,
      "budget": 10000,
      "budget_utilization": 1.0,
      "success_levels": {
        "gold": {
          "success": false,
          "threshold": 1.0,
          "hit_evaluations": null
        },
        "silver": {
          "success": false,
          "threshold": 5.0,
          "hit_evaluations": null
        },
        "bronze": {
          "success": true,
          "threshold": 10.0,
          "hit_evaluations": 8200
        }
      },
      "status": "ok",
      "error_type": null,
      "error_msg": null
    }
  ]
}
```

**Highlights:**

* **Metadata**: Centralized configuration tracking, status breakdown, budget utilization metrics
* **Tracking**: Every run has `status`, `error_type`, `error_msg` for error investigation
* **Hit time**: `hit_evaluations` records when target threshold was achieved (null if never hit)
* **Budget control**: `budget_utilization` ensures algorithm stayed within evaluation budget
* **Problem seed**: `problem_seed` enables problem reproducibility

### Knapsack Results

**File naming:** `benchmark/results/knapsack/knapsack_n{size}_{type}_seed{seed}_{algo}_{strategy}_{timestamp}.json`

```json
{
  "metadata": {
    "problem": "knapsack",
    "n_items": 50,
    "instance_type": "uncorrelated",
    "instance_seed": 42,
    "algorithm": "FA",
    "timestamp": "20251110T202419",
    "capacity": 500.0,
    "budget": 10000,
    "n_runs": 30,
    "n_successful": 30,
    "n_failed": 0,
    "status_breakdown": {"ok": 30},
    "dp_optimal": 2450.0,
    "gap_thresholds": {
      "gold": 1.0,
      "silver": 5.0,
      "bronze": 10.0
    },
    "tier_success_rates": {
      "SR_Gold_%": 5.0,
      "SR_Silver_%": 45.0,
      "SR_Bronze_%": 80.0
    },
    "constraint_handling": "repair",
    "avg_gap_%": 2.34,
    "avg_feasibility_rate": 1.0
  },
  "all_results": [
    {
      "algorithm": "FA",
      "seed": 0,
      "algo_seed": 0,
      "instance_seed": 42,
      "best_value": 2387.0,
      "best_fitness": -2387.0,
      "total_weight": 487.5,
      "is_feasible": true,
      "gap_relative": 2.57,
      "gap_tier": "silver",
      "success_levels": {
        "gold": {
          "success": false,
          "threshold": 1.0,
          "hit_evaluations": null
        },
        "silver": {
          "success": true,
          "threshold": 5.0,
          "hit_evaluations": 8200
        },
        "bronze": {
          "success": true,
          "threshold": 10.0,
          "hit_evaluations": 3400
        }
      },
      "history": [-1200.0, -1500.0, ..., -2387.0],
      "elapsed_time": 3.45,
      "items_selected": 18,
      "capacity_utilization": 0.975,
      "status": "ok"
    }
  ]
}
```

**Key Updates:**
- **gap_thresholds**: Multi-tier thresholds (Gold/Silver/Bronze)
- **tier_success_rates**: Success rate at each tier
- **gap_relative**: Relative gap (%) to DP optimal
- **gap_tier**: Best tier achieved ("gold"/"silver"/"bronze"/null)
- **success_levels**: Per-tier hit time tracking (like Rastrigin)

