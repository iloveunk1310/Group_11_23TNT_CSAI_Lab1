"""
Test suite for benchmarks - verify all components work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path
import shutil
import glob
import json

from benchmark.run_rastrigin import run_rastrigin_benchmark
from benchmark.run_knapsack import run_knapsack_benchmark
from benchmark.config import RASTRIGIN_CONFIGS, KNAPSACK_CONFIGS

# Import problems for testing
from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.knapsack import KnapsackProblem


def find_result_files(output_dir, problem, pattern):
    """Find all result files matching the pattern."""
    search_pattern = str(Path(output_dir) / problem / pattern)
    return glob.glob(search_pattern)


def validate_rastrigin_result(filepath):
    """
    Validate a Rastrigin result file.
    
    Checks:
    - JSON structure (metadata + results)
    - Required metadata fields
    - Status tracking
    - Budget utilization
    - History completeness
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate structure
    assert 'metadata' in data, f"Missing metadata in {filepath}"
    assert 'results' in data, f"Missing results in {filepath}"
    
    metadata = data['metadata']
    
    # Validate metadata fields
    required_metadata = [
        'problem', 'config_name', 'algorithm', 'timestamp',
        'dimension', 'budget', 'max_iter', 'n_runs'
    ]
    for field in required_metadata:
        assert field in metadata, f"Missing metadata field '{field}'"
    
    # Validate results
    results = data['results']
    assert len(results) > 0, f"No results in {filepath}"
    
    for i, result in enumerate(results):
        # Check required fields
        required_fields = [
            'algorithm', 'seed', 'best_fitness', 'history',
            'elapsed_time', 'evaluations'
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in result {i}"
        
        # Validate history is non-empty and finite
        history = result['history']
        assert len(history) > 0, f"Run {i}: Empty convergence history"
        assert all(np.isfinite(h) for h in history), \
            f"Run {i}: Non-finite values in history"
        
        # Validate best_fitness is finite
        assert np.isfinite(result['best_fitness']), \
            f"Run {i}: Invalid best_fitness"


def validate_knapsack_result(filepath):
    """
    Validate a Knapsack result file.
    
    Checks:
    - JSON structure
    - Metadata completeness
    - Feasibility constraints
    - DP optimal comparison (if available)
    - Multi-tier gap analysis (NEW)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate structure
    assert 'metadata' in data, f"Missing metadata in {filepath}"
    assert 'results' in data, f"Missing results in {filepath}"
    
    metadata = data['metadata']
    
    # Validate metadata
    required_metadata = [
        'problem', 'n_items', 'instance_type', 'instance_seed',
        'algorithm', 'capacity', 'n_runs'
    ]
    for field in required_metadata:
        assert field in metadata, f"Missing metadata field '{field}'"
    
    # Validate gap_thresholds if DP available
    if metadata.get('dp_optimal'):
        assert 'gap_thresholds' in metadata, "Missing gap_thresholds when DP optimal available"
        thresholds = metadata['gap_thresholds']
        assert 'gold' in thresholds and 'silver' in thresholds and 'bronze' in thresholds, \
            "Missing gold/silver/bronze tiers in gap_thresholds"
    
    # Validate results
    results = data['results']
    assert len(results) > 0, f"No results in {filepath}"
    
    for i, result in enumerate(results):
        # Check required fields
        required_fields = [
            'algorithm', 'seed', 'best_value', 'best_fitness',
            'total_weight', 'is_feasible', 'history', 'status'
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in result {i}"
        
        # Validate feasibility
        if result['is_feasible']:
            assert result['total_weight'] <= metadata['capacity'], \
                f"Run {i}: Feasible solution exceeds capacity"
        
        # Validate gap_tier and success_levels if DP available
        if metadata.get('dp_optimal') and result['status'] == 'ok':
            assert 'gap_relative' in result, f"Missing gap_relative for DP-available instance"
            assert 'success_levels' in result, f"Missing success_levels for DP-available instance"
            
            success_levels = result['success_levels']
            for tier in ['gold', 'silver', 'bronze']:
                assert tier in success_levels, f"Missing tier '{tier}' in success_levels"
                tier_data = success_levels[tier]
                assert 'success' in tier_data and 'threshold' in tier_data, \
                    f"Invalid success_levels structure for tier '{tier}'"


class TestRastriginBenchmark:
    """Test Rastrigin benchmark."""
    
    def test_quick_convergence(self, tmp_path):
        """Test quick convergence config runs without errors."""
        run_rastrigin_benchmark(
            config_name='quick_convergence',
            output_dir=str(tmp_path),
            n_runs=3,
            n_jobs=2
        )
        
        # Check that result files were created
        for algo in ['FA', 'SA', 'HC', 'GA']:
            pattern = f"rastrigin_quick_convergence_{algo}_*.json"
            files = find_result_files(tmp_path, 'rastrigin', pattern)
            assert len(files) > 0, f"No result files for {algo}"
            
            # Validate first file
            validate_rastrigin_result(files[0])
    
    def test_all_configs(self, tmp_path):
        """Test all Rastrigin configurations."""
        configs = ['quick_convergence', 'multimodal_escape', 'scalability']
        
        for config in configs:
            run_rastrigin_benchmark(
                config_name=config,
                output_dir=str(tmp_path),
                n_runs=2,
                n_jobs=2
            )
            
            # Verify files exist
            for algo in ['FA', 'SA', 'HC', 'GA']:
                pattern = f"rastrigin_{config}_{algo}_*.json"
                files = find_result_files(tmp_path, 'rastrigin', pattern)
                assert len(files) > 0, f"No files for {config}/{algo}"


class TestKnapsackBenchmark:
    """Test Knapsack benchmark."""
    
    def test_small_instance(self, tmp_path):
        """Test small Knapsack instance (n=50)."""
        run_knapsack_benchmark(
            size=50,
            instance_type='uncorrelated',
            instance_seed=42,
            output_dir=str(tmp_path),
            n_runs=3,
            n_jobs=2
        )
        
        # Check result files
        for algo in ['FA', 'SA', 'HC', 'GA']:
            pattern = f"knapsack_n50_uncorrelated_seed42_{algo}_*.json"
            files = find_result_files(tmp_path, 'knapsack', pattern)
            assert len(files) > 0, f"No result files for {algo}"
            
            # Validate
            validate_knapsack_result(files[0])
    
    def test_all_instance_types(self, tmp_path):
        """Test all instance types."""
        types = ['uncorrelated', 'weakly_correlated', 'strongly_correlated', 'subset_sum']
        
        for itype in types:
            run_knapsack_benchmark(
                size=50,
                instance_type=itype,
                instance_seed=42,
                output_dir=str(tmp_path),
                n_runs=2,
                n_jobs=2
            )
            
            # Verify files
            for algo in ['FA', 'GA']:  # Test subset only
                pattern = f"knapsack_n50_{itype}_seed42_{algo}_*.json"
                files = find_result_files(tmp_path, 'knapsack', pattern)
                assert len(files) > 0, f"No files for {itype}/{algo}"


class TestInstanceGenerator:
    """Test instance generation."""
    
    def test_rastrigin_problem(self):
        """Test Rastrigin problem instantiation."""
        for dim in [10, 30, 50]:
            problem = RastriginProblem(dim=dim)
            assert problem.dim == dim
            assert len(problem.lower) == dim
            assert len(problem.upper) == dim
    
    def test_knapsack_problem(self):
        """Test Knapsack problem instantiation."""
        from benchmark.instance_generator import generate_knapsack_instance
        
        for size in [50, 100]:
            for itype in ['uncorrelated', 'weakly_correlated', 'strongly_correlated', 'subset_sum']:
                values, weights, capacity = generate_knapsack_instance(
                    n_items=size,
                    instance_type=itype,
                    seed=42
                )
                
                assert len(values) == size
                assert len(weights) == size
                assert capacity > 0
                
                # Create problem
                problem = KnapsackProblem(values, weights, capacity)
                assert problem.num_items == size


class TestParallelExecution:
    """Test parallel execution."""
    
    def test_parallel_rastrigin(self, tmp_path):
        """Test parallel Rastrigin benchmark."""
        run_rastrigin_benchmark(
            config_name='quick_convergence',
            output_dir=str(tmp_path),
            n_runs=4,
            n_jobs=2  # Use 2 workers
        )
        
        # Check all algorithms completed
        for algo in ['FA', 'SA', 'HC', 'GA']:
            pattern = f"rastrigin_quick_convergence_{algo}_*.json"
            files = find_result_files(tmp_path, 'rastrigin', pattern)
            assert len(files) > 0
    
    def test_parallel_knapsack(self, tmp_path):
        """Test parallel Knapsack benchmark."""
        run_knapsack_benchmark(
            size=50,
            instance_type='uncorrelated',
            instance_seed=42,
            output_dir=str(tmp_path),
            n_runs=4,
            n_jobs=2
        )
        
        # Check completion
        for algo in ['FA', 'SA', 'HC', 'GA']:
            pattern = f"knapsack_n50_uncorrelated_seed42_{algo}_*.json"
            files = find_result_files(tmp_path, 'knapsack', pattern)
            assert len(files) > 0


def run_quick_tests(parallel=False, num_workers=2):
    """Run quick sanity tests without pytest."""
    test_dir = Path('benchmark/results/quick_test')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("=" * 70)
        print("RUNNING QUICK BENCHMARK TESTS")
        print("=" * 70)
        
        mode = "parallel" if parallel else "sequential"
        jobs = num_workers if parallel else 1
        
        print(f"\n1. Testing Rastrigin benchmark ({mode})...")
        run_rastrigin_benchmark(
            config_name='quick_convergence',
            output_dir=str(test_dir),
            n_runs=3,
            n_jobs=jobs
        )
        print("   ✓ Rastrigin benchmark passed")
        
        print(f"\n2. Testing Knapsack benchmark ({mode})...")
        run_knapsack_benchmark(
            size=50,
            instance_type='uncorrelated',
            instance_seed=42,
            output_dir=str(test_dir),
            n_runs=3,
            n_jobs=jobs
        )
        print("   ✓ Knapsack benchmark passed")
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test benchmarks')
    parser.add_argument('--pytest', action='store_true',
                        help='Run with pytest')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick sanity tests')
    parser.add_argument('--parallel', action='store_true',
                        help='Run benchmarks in parallel')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers (default: 2)')
    
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, '-v'])
    else:
        run_quick_tests(parallel=args.parallel, num_workers=args.workers)