# Quick Start Guide - AI Optimization Framework

## Installation & Setup

```bash
cd /home/bui-anh-quan/Firefly

# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate firefly
```

## Quick Demo

```bash
# Quick demo with parallel execution (4 cores)
python demo.py --parallel --jobs 4

# Or run specific benchmarks
python benchmark/run_rastrigin.py --config quick_convergence --jobs 4
python benchmark/run_knapsack.py --size 50 --jobs 4
```

## ⏱️ Estimated Runtime (with 4 cores)

| Benchmark           | Sequential | Parallel (4 cores) |
| ------------------- | ---------- | ------------------ |
| Rastrigin quick     | ~5 min     | ~2 min             |
| Rastrigin all       | ~45 min    | ~15 min            |
| Knapsack n=50       | ~30 min    | ~10 min            |
| Knapsack n=100      | ~1 hour    | ~20 min            |
| Analysis + Plots    | ~10 min    | ~10 min            |
| **Total Full Suite** | **~7 hours** | **~2-3 hours**     |

## New Features in Analysis

### 1. Multi-Tier Success Analysis

Both problems now track success at multiple difficulty levels:

**Rastrigin:**
```python
# Results now include multi-tier tracking
{
  "success_levels": {
    "gold": {"success": false, "threshold": 1.0, "hit_evaluations": null},
    "silver": {"success": true, "threshold": 10.0, "hit_evaluations": 8200},
    "bronze": {"success": true, "threshold": 30.0, "hit_evaluations": 3400}
  }
}
```

**Knapsack:**
```python
# Gap-based multi-tier tracking
{
  "gap_relative": 2.57,  # % gap from optimal
  "gap_tier": "silver",  # Best tier achieved
  "success_levels": {
    "gold": {"success": false, "threshold": 1.0, "hit_evaluations": null},
    "silver": {"success": true, "threshold": 5.0, "hit_evaluations": 8200},
    "bronze": {"success": true, "threshold": 10.0, "hit_evaluations": 3400}
  }
}
```

### 2. COCO/BBOB-Standard Analysis

New analysis outputs from `analyze_results.py`:

```bash
# Generate all COCO/BBOB metrics
python benchmark/analyze_results.py --problem all

# Output artifacts:
# - Fixed-target ECDF (runtime-to-target distributions)
# - Expected Running Time (ERT) with bootstrap CIs
# - Performance Profiles (Dolan-Moré)
# - Data Profiles (Moré-Wild)
# - Fixed-budget checkpoints
```

### 3. Enhanced Visualizations

```bash
# Generate all plots
python benchmark/visualize.py

# New plot types:
# - ECDF curves (per tier, per config)
# - ERT bar charts with error bars
# - Performance profiles (robustness)
# - Data profiles (budget-quality)
# - Diversity panels (Rastrigin only)
# - Stagnation analysis
# - Pairwise heatmaps (Knapsack)
# - Copeland rankings
```

## Testing Your Implementation

### 1. Unit Tests

```bash
# Run all tests
python test/run_all_tests.py

# Test specific components
python -m unittest test.test_problems
python -m unittest test.test_firefly_algorithm
python -m unittest test.test_utils
```

### 2. Quick Examples

#### Example 1: Run FA on Rastrigin with Multi-Tier Tracking

```python
import sys
sys.path.append('/home/bui-anh-quan/Firefly')

from src.problems.continuous.rastrigin import RastriginProblem
from src.swarm.fa import FireflyContinuousOptimizer

# Setup
problem = RastriginProblem(dim=10)
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=40,
    alpha=0.3,
    seed=42
)

# Run with multi-tier tracking
thresholds = {'gold': 1.0, 'silver': 10.0, 'bronze': 30.0}
best_sol, best_fit, history, stats = optimizer.run(
    max_iter=250,
    target_thresholds=thresholds  # NEW: Multi-tier tracking
)

# Check which tiers were achieved
# (This is tracked internally and saved in results JSON)
print(f"Best fitness: {best_fit:.6f}")
print(f"Convergence: {history[0]:.4f} -> {history[-1]:.4f}")
```

#### Example 2: Knapsack with Corrected Gap Analysis

```python
import sys
import numpy as np
sys.path.append('/home/bui-anh-quan/Firefly')

from src.problems.discrete.knapsack import KnapsackProblem
from src.swarm.fa import FireflyKnapsackOptimizer

# Create instance
rng = np.random.RandomState(42)
n_items = 50
values = rng.randint(10, 100, n_items)
weights = rng.randint(1, 50, n_items)
capacity = int(0.5 * np.sum(weights))

problem = KnapsackProblem(values, weights, capacity)
dp_optimal = problem.solve_dp()

# Run optimization
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=60,
    constraint_handling="repair",  # Fair comparison
    seed=42
)

best_sol, best_fit, history, _ = optimizer.run(max_iter=166)

# Compute gap correctly (for maximization)
best_value = -best_fit  # Negate fitness to get value
gap = 100.0 * (dp_optimal - best_value) / dp_optimal

print(f"DP Optimal: {dp_optimal:.0f}")
print(f"Best value: {best_value:.0f}")
print(f"Gap: {gap:.2f}%")

# Check which tier achieved
if gap <= 1.0:
    print("Tier: GOLD 🥇")
elif gap <= 5.0:
    print("Tier: SILVER 🥈")
elif gap <= 10.0:
    print("Tier: BRONZE 🥉")
else:
    print("Tier: None")
```

#### Example 3: Load and Analyze Results

```python
import sys
import json
import gzip
sys.path.append('/home/bui-anh-quan/Firefly')

from benchmark.analyze_results import load_all_results_to_dataframe

# Load all results into unified DataFrame
df = load_all_results_to_dataframe('benchmark/results')

# Filter successful runs
df_ok = df[df['Status'] == 'ok']

# Rastrigin: Check multi-tier success
df_rast = df_ok[df_ok['Problem'] == 'rastrigin']
for level in ['Gold', 'Silver', 'Bronze']:
    success_col = f'Success_{level}'
    if success_col in df_rast.columns:
        sr = df_rast[success_col].mean() * 100
        print(f"Rastrigin {level} Success Rate: {sr:.1f}%")

# Knapsack: Check gap distribution
df_knap = df_ok[df_ok['Problem'] == 'knapsack']
if 'Optimality_Gap' in df_knap.columns:
    gaps = df_knap['Optimality_Gap'].dropna()
    print(f"\nKnapsack Gap Distribution:")
    print(f"  Median: {gaps.median():.2f}%")
    print(f"  Q25-Q75: {gaps.quantile(0.25):.2f}% - {gaps.quantile(0.75):.2f}%")
    print(f"  Gold (<1%): {(gaps <= 1.0).mean()*100:.1f}%")
    print(f"  Silver (<5%): {(gaps <= 5.0).mean()*100:.1f}%")
    print(f"  Bronze (<10%): {(gaps <= 10.0).mean()*100:.1f}%")
```

## Creating Visualizations

### Option 1: Use Built-in Visualization Functions

```python
import sys
sys.path.append('/home/bui-anh-quan/Firefly')

from benchmark.visualize import (
    plot_rastrigin_fixed_target_ecdf,
    plot_rastrigin_ert,
    plot_knapsack_performance_profiles_dolan_more,
    load_summary
)
from pathlib import Path

summary_dir = Path('benchmark/results/summaries')
output_dir = Path('benchmark/results/plots')

# Rastrigin ECDF
ecdf_df = load_summary('rastrigin_fixed_target_ecdf', summary_dir)
if ecdf_df is not None:
    plot_rastrigin_fixed_target_ecdf(ecdf_df, output_dir)

# Rastrigin ERT
ert_df = load_summary('rastrigin_ert', summary_dir)
if ert_df is not None:
    plot_rastrigin_ert(ert_df, output_dir, level='Silver')

# Knapsack Performance Profiles
perf_df = load_summary('knapsack_performance_profiles', summary_dir)
if perf_df is not None:
    plot_knapsack_performance_profiles_dolan_more(perf_df, output_dir)
```

### Option 2: Generate All Plots at Once

```bash
# Generate all visualizations from CSV summaries
python benchmark/visualize.py

# Or for specific problem
python benchmark/visualize.py --problem rastrigin
python benchmark/visualize.py --problem knapsack
```

## Understanding New Metrics

### 1. Expected Running Time (ERT)

- **Definition**: Average evaluations needed to reach target
- **Includes failures**: Failed runs use full budget (censored analysis)
- **Lower is better**: Fewer evaluations = faster convergence
- **With confidence intervals**: Bootstrap 95% CI

### 2. Performance Profiles (Dolan-Moré)

- **X-axis**: τ = performance ratio (your_time / best_time)
- **Y-axis**: φ(τ) = fraction of problems solved within τ × best
- **Higher curve = more robust**: Reaches 100% at smaller τ

### 3. Data Profiles (Moré-Wild)

- **X-axis**: ν = budget (evaluations)
- **Y-axis**: ψ(ν) = fraction of problems solved within budget ν
- **Steeper = faster**: Quickly solves more problems

### 4. Fixed-Target ECDF

- **X-axis**: Runtime (evaluations to hit target, log scale)
- **Y-axis**: ECDF = cumulative fraction of runs hitting target
- **Leftward shift = faster**: Less time to reach target

### 5. Diversity Metrics (Rastrigin)

- **Normalized by √D**: Allows fair comparison across dimensions
- **Initial/Mid/Final**: Track diversity evolution
- **Drop**: Total diversity loss (high = premature convergence)

## Parameter Tuning Guide

### Firefly Algorithm (Continuous)

```python
optimizer = FireflyContinuousOptimizer(
    problem=problem,
    n_fireflies=40,   # 20-50 typical, higher for multimodal
    alpha=0.3,        # 0.1-0.5, higher for more exploration
    beta0=1.0,        # 0.5-2.0, base attractiveness
    gamma=1.0,        # 0.1-2.0, lower for global search
    seed=42
)
```

**For multimodal problems (Rastrigin):**
- Increase `alpha` (0.3-0.5) for more exploration
- Decrease `gamma` (0.5-1.0) for long-range attraction
- Increase `n_fireflies` (30-50) for better coverage

### Firefly Algorithm (Knapsack)

```python
optimizer = FireflyKnapsackOptimizer(
    problem=problem,
    n_fireflies=60,           # 40-80 typical
    alpha_flip=0.2,           # 0.1-0.4, random bit flip prob
    max_flips_per_move=3,     # 2-5, directed flips
    constraint_handling="repair",  # "repair" or "penalty"
    seed=42
)
```

**Constraint handling:**
- **`repair`**: Greedy repair → all solutions feasible → **fair comparison**
- **`penalty`**: Large penalty → may explore infeasible space

## Common Issues & Solutions

### Issue 1: ECDF is empty for Knapsack

**Symptom**: `build_fixed_target_ecdf` returns empty DataFrame

**Cause**: `hit_evaluations` in JSON is `null` (not pre-computed)

**Solution**: The new code computes runtime from history on-the-fly:
```python
# Now handles automatically in analyze_results.py
df_ecdf = build_fixed_target_ecdf(df, 'knapsack')
# No longer relies on null hit_evaluations field
```

### Issue 2: Gap values seem too large

**Symptom**: Most gaps are >100% or negative

**Cause**: Sign error in gap computation (previous version)

**Solution**: Now correctly computes gap for maximization:
```python
# CORRECT (current version)
gap = 100.0 * (optimal - achieved) / optimal

# Gap = 0% means optimal
# Gap = 5% means achieved is 95% of optimal
```

### Issue 3: Data profiles show zero success

**Symptom**: All `psi(ν)` values are 0

**Cause**: History contains fitness (negative) instead of value (positive)

**Solution**: Now correctly converts history:
```python
# In analyze_results.py, now handles sign conversion:
if best_value > 0 and best_fitness < 0:
    values = [-v for v in history]  # Convert to positive
```

### Issue 4: Performance profiles have infinite ratios

**Symptom**: All algorithms show ratio = ∞

**Cause**: No algorithm successfully solves any instance

**Solution**: Check if targets are too strict, or increase budget:
```bash
# Use more lenient targets (Bronze tier)
# Or increase budget in config.py
```

## Next Steps

1. **Run full benchmark**: `python benchmark/run_all.py --full --jobs 4`
2. **Generate analysis**: `python benchmark/analyze_results.py --problem all`
3. **Create plots**: `python benchmark/visualize.py`
4. **Review CSVs**: Check `benchmark/results/summaries/*.csv`
5. **Read papers**: See references in main README.md

## File Structure Summary

```
benchmark/
├── run_rastrigin.py       # Rastrigin runner with multi-tier tracking
├── run_knapsack.py        # Knapsack runner with gap-based tracking
├── analyze_results.py     # COCO/BBOB analysis (NEW: corrected gaps)
├── visualize.py           # Academic visualizations (NEW: profiles)
├── config.py              # Centralized configs (multi-tier thresholds)
└── results/
    ├── rastrigin/         # Raw JSON.GZ files
    ├── knapsack/          # Raw JSON.GZ files
    ├── summaries/         # CSV analysis outputs (NEW: many types)
    │   ├── rastrigin_fixed_target_ecdf.csv
    │   ├── rastrigin_ert.csv
    │   ├── rastrigin_performance_profiles.csv
    │   ├── knapsack_fixed_target_ecdf.csv
    │   └── ...
    └── plots/             # All visualizations (NEW: COCO standard)
        ├── rastrigin_ecdf_quick_convergence.png
        ├── rastrigin_ert_silver.png
        ├── rastrigin_perf_profile.png
        ├── knapsack_performance_profiles.png
        └── ...
```

## Key Concepts

### All algorithms return the same format:
```python
best_solution, best_fitness, history, trajectory = optimizer.run(max_iter)
```

- `history`: List of best fitness per iteration (for convergence plots)
- `trajectory`: List of populations per iteration (for animations)
- Multi-tier tracking happens internally and is saved in JSON

### All analysis now uses unified DataFrame:
```python
from benchmark.analyze_results import load_all_results_to_dataframe

df = load_all_results_to_dataframe('benchmark/results')
# Automatically handles JSON and JSON.GZ
# Extracts multi-tier success_levels
# Computes runtime-to-target from history
# Handles both Rastrigin and Knapsack
```

This allows **consistent analysis** across problems and algorithms!
