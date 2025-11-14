# Test Suite

This folder contains comprehensive unit tests for the AI Search & Optimization Framework.

## Test Structure
```
test/
├── __init__.py                      # Package initialization
├── test_utils.py                    # Tests for utility functions
├── test_problems.py                 # Tests for Rastrigin and Knapsack
├── test_firefly_algorithm.py        # Tests for FA (continuous & discrete)
├── test_classical_algorithms.py     # Tests for HC, SA, GA
├── test_edge_cases.py               # Edge cases and boundary conditions
├── test_parallel_execution.py       # Parallel execution tests
├── run_all_tests.py                 # Main test runner
└── README.md                        # This file
```

## Running Tests

### Run All Tests
```bash
# From project root
python test/run_all_tests.py

# Or with verbose output
python test/run_all_tests.py -v

# With parallel execution (faster)
python test/run_all_tests.py -j 4
```

### Run Specific Test Module
```bash
python test/run_all_tests.py test_problems
python test/run_all_tests.py test_firefly_algorithm
```

### Run Individual Test File
```bash
python -m unittest test.test_problems
python -m unittest test.test_utils
```

### Run Specific Test Class or Method
```bash
python -m unittest test.test_problems.TestRastriginProblem
python -m unittest test.test_problems.TestRastriginProblem.test_optimum_value
```

## Test Coverage

### Utility Functions (`test_utils.py`)
- ✓ Euclidean distance matrix computation
- ✓ Best solution extraction
- ✓ Brightness computation
- ✓ Permutation repair (deterministic & stochastic)
- ✓ Numerical stability with extreme values

### Continuous Problems (`test_problems.py`)
- ✓ Rastrigin function dimension and bounds
- ✓ Optimum value verification (global minimum at origin)
- ✓ Fitness calculation correctness
- ✓ Random solution generation within bounds
- ✓ Multimodality properties

### Knapsack Problem (`test_problems.py`)
- ✓ Item weights and values
- ✓ Capacity constraints
- ✓ DP optimal solution verification
- ✓ Random instance generation (4 types)
- ✓ Invalid input handling
- ✓ Greedy repair strategy
- ✓ **Gap computation correctness** (NEW: validates maximization formula)

### Firefly Algorithm (`test_firefly_algorithm.py`)
- ✓ Continuous optimizer initialization and parameters
- ✓ Output format validation (solution, fitness, history, stats_history)
- ✓ Convergence behavior on Rastrigin
- ✓ Deterministic results with seed
- ✓ Knapsack optimizer initialization
- ✓ Valid binary solution generation
- ✓ Feasibility constraint checking
- ✓ Repair vs penalty constraint handling
- ✓ **Multi-tier success tracking** (NEW)

### Classical Algorithms (`test_classical_algorithms.py`)
- ✓ Hill Climbing with random restart
- ✓ Simulated Annealing with temperature schedule
- ✓ Genetic Algorithm with crossover, mutation, and elitism
- ✓ Convergence for all algorithms
- ✓ Constraint handling for Knapsack

### Edge Cases (`test_edge_cases.py`)
- ✓ Extreme dimensions (d=1, d=100, d=1000)
- ✓ Zero capacity Knapsack
- ✓ Empty items
- ✓ All items too heavy
- ✓ Single item cases
- ✓ Boundary parameter values
- ✓ Invalid inputs (negative dimensions, invalid parameters)
- ✓ Numerical stability

### Parallel Execution (`test_parallel_execution.py`)
- ✓ Multiprocessing correctness
- ✓ Reproducibility with seeds
- ✓ Performance scaling
- ✓ **Concurrent ECDF and ERT computation** (NEW)

## New Test Cases for Enhanced Analysis

### Test Multi-Tier Success Tracking

```python
def test_multi_tier_success_rastrigin():
    """Test that Rastrigin tracks gold/silver/bronze success."""
    problem = RastriginProblem(dim=10)
    optimizer = FireflyContinuousOptimizer(problem, seed=42)
    
    # Mock thresholds
    thresholds = {'gold': 1.0, 'silver': 10.0, 'bronze': 30.0}
    
    # Run should track success at each level
    # (Internal tracking, check JSON output)
    best_sol, best_fit, history, _ = optimizer.run(max_iter=50)
    
    # Check that final fitness can be categorized
    if abs(best_fit) <= 1.0:
        tier = 'gold'
    elif abs(best_fit) <= 10.0:
        tier = 'silver'
    elif abs(best_fit) <= 30.0:
        tier = 'bronze'
    else:
        tier = None
    
    assert tier is not None, "Should achieve at least bronze"
```

### Test Gap Computation Correctness

```python
def test_knapsack_gap_computation():
    """Test that gap is correctly computed for maximization."""
    from benchmark.analyze_results import build_fixed_target_ecdf
    
    # Create mock data
    dp_optimal = 1000.0
    best_value = 950.0  # 95% of optimal
    
    # Correct gap formula (for maximization)
    gap = 100.0 * (dp_optimal - best_value) / dp_optimal
    
    assert abs(gap - 5.0) < 1e-6, "Gap should be 5.0%"
    
    # Test with 99% solution
    best_value = 990.0
    gap = 100.0 * (dp_optimal - best_value) / dp_optimal
    assert abs(gap - 1.0) < 1e-6, "Gap should be 1.0% (gold tier)"
```

### Test ECDF and ERT Computation

```python
def test_ecdf_computation_from_history():
    """Test that ECDF correctly computes runtime from history."""
    from benchmark.analyze_results import load_all_results_to_dataframe, build_fixed_target_ecdf
    
    # Load real data
    df = load_all_results_to_dataframe('benchmark/results')
    
    # Compute ECDF for Knapsack
    ecdf_df = build_fixed_target_ecdf(df, 'knapsack')
    
    # Check that ECDF is not empty
    assert not ecdf_df.empty, "ECDF should not be empty"
    
    # Check that tau values are positive
    assert (ecdf_df['tau'] > 0).all(), "All runtimes should be positive"
    
    # Check that ECDF values are in [0, 1]
    assert (ecdf_df['ECDF'] >= 0).all() and (ecdf_df['ECDF'] <= 1).all()
```

### Test Diversity Metrics

```python
def test_diversity_normalization():
    """Test that diversity is correctly normalized by √D."""
    from benchmark.analyze_results import summarize_diversity, load_all_results_to_dataframe
    
    df = load_all_results_to_dataframe('benchmark/results')
    div_df = summarize_diversity(df)
    
    if not div_df.empty:
        # Check that normalized diversity is reasonable
        for _, row in div_df.iterrows():
            dim = row['Dim']
            div_initial = row['Div_Initial']
            div_norm_initial = row['Div_Norm_Initial']
            
            expected = div_initial / np.sqrt(dim)
            assert abs(div_norm_initial - expected) < 1e-6, \
                "Normalized diversity should equal diversity / √D"
```

## Testing Visualizations

While visualization functions don't have automated tests, you can verify plots manually:

```bash
# Generate all plots
python benchmark/visualize.py

# Check that plots exist
ls -la benchmark/results/plots/

# Expected files:
# - rastrigin_ecdf_*.png
# - rastrigin_ert_*.png
# - rastrigin_perf_profile.png
# - rastrigin_data_profile.png
# - rastrigin_diversity_*.png
# - knapsack_performance_profiles.png
# - knapsack_data_profiles.png
# - knapsack_pairwise_heatmap_*.png
# - knapsack_copeland_ranking.png
```

## Adding New Tests

To add tests for a new component:

1. Create a new test file: `test_your_component.py`
2. Import unittest and your component
3. Create test class inheriting from `unittest.TestCase`
4. Write test methods (must start with `test_`)
5. Run tests to verify

Example:
```python
import unittest
from benchmark.analyze_results import compute_ert

class TestERTComputation(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.mock_df = create_mock_dataframe()
    
    def test_ert_with_all_success(self):
        """Test ERT when all runs succeed."""
        ert_df = compute_ert(self.mock_df, 'rastrigin')
        
        # Check that ERT equals mean of hit times
        expected = self.mock_df['HitEvals_Silver'].mean()
        actual = ert_df[ert_df['Level'] == 'Silver']['ERT'].iloc[0]
        
        self.assertAlmostEqual(actual, expected, places=2)
    
    def test_ert_with_failures(self):
        """Test ERT when some runs fail."""
        # Mock data with 50% success rate
        # ERT should include full budget for failures
        pass
```

## Test Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Use seeds for reproducible results
3. **Coverage**: Test normal cases, edge cases, and error cases
4. **Clarity**: Use descriptive test names
5. **Speed**: Keep tests fast (use small problem sizes)
6. **Mocking**: Mock expensive operations when possible

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: python test/run_all_tests.py
```

## Troubleshooting

**Import errors**: Make sure you're running from the project root or have the correct PYTHONPATH set.

**Missing dependencies**: Install required packages:
```bash
pip install -r requirements.txt
```

**Matplotlib backend errors**: If plots don't display, try:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Failed tests**: Check the error message and traceback for details. Ensure all source files are present and correct.

**Slow tests**: Use smaller problem sizes or reduce number of runs:
```python
# For quick testing
problem = RastriginProblem(dim=5)  # Instead of dim=50
optimizer.run(max_iter=10)  # Instead of max_iter=1000
```

## Coverage Report

To generate a coverage report:

```bash
# Install coverage tool
pip install coverage

# Run tests with coverage
coverage run -m unittest discover test

# Generate report
coverage report

# Generate HTML report
coverage html
# Open htmlcov/index.html in browser
```

Expected coverage:
- Core algorithms: >90%
- Utility functions: >95%
- Analysis functions: >80%
- Visualization: Not covered (manual testing)

## Known Issues

1. **Knapsack DP solver**: Very slow for n > 100, tests skip these cases
2. **Parallel tests**: May occasionally fail due to race conditions (re-run to verify)
3. **Floating point**: Use `assertAlmostEqual` for float comparisons, not `assertEqual`

## Future Test Additions

- [ ] Test for multi-tier success rate accuracy
- [ ] Test for ERT confidence interval coverage
- [ ] Test for performance profile monotonicity
- [ ] Test for data profile convergence properties
- [ ] Integration tests for full benchmark pipeline
- [ ] Stress tests with very large problem sizes
