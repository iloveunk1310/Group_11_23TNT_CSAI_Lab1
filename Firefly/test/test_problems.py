"""
Unit tests for optimization problems (continuous and discrete).
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.knapsack import KnapsackProblem


class TestRastriginProblem(unittest.TestCase):
    """Test cases for Rastrigin function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = RastriginProblem(dim=3)
    
    def test_dimension(self):
        """Test problem dimension."""
        self.assertEqual(self.problem.dim, 3)
    
    def test_bounds(self):
        """Test problem bounds."""
        self.assertEqual(len(self.problem.lower), 3)
        self.assertEqual(len(self.problem.upper), 3)
        self.assertTrue(all(self.problem.lower == -5.12))
        self.assertTrue(all(self.problem.upper == 5.12))
    
    def test_optimum_value(self):
        """Test that optimum (origin) gives zero fitness."""
        optimum = np.zeros(3)
        fitness = self.problem.evaluate(optimum)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_fitness_positive(self):
        """Test that fitness is always non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.randn(3) * 5
            fitness = self.problem.evaluate(x)
            self.assertGreaterEqual(fitness, 0.0)
    
    def test_random_solution(self):
        """Test random solution generation."""
        rng = np.random.RandomState(42)
        solutions = self.problem.init_solution(rng, n=1)
        solution = solutions[0]
        self.assertEqual(len(solution), 3)
        self.assertTrue(all(solution >= -5.12))
        self.assertTrue(all(solution <= 5.12))
    
    def test_multimodality(self):
        """Test that Rastrigin is multimodal (has local minima)."""
        x_local = np.array([0.99, 0.99, 0.99])
        f_local = self.problem.evaluate(x_local)
        
        self.assertGreater(f_local, 0.0)
        self.assertLess(f_local, 10.0)
    
    def test_representation_type(self):
        """Test that representation type is continuous."""
        self.assertEqual(self.problem.representation_type(), "continuous")


class TestKnapsackProblem(unittest.TestCase):
    """Test cases for Knapsack problem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.values = np.array([60, 100, 120])
        self.weights = np.array([10, 20, 30])
        self.capacity = 50.0
        self.problem = KnapsackProblem(self.values, self.weights, self.capacity)
    
    def test_initialization(self):
        """Test problem initialization."""
        self.assertEqual(self.problem.num_items, 3)
        self.assertEqual(self.problem.capacity, 50.0)
        np.testing.assert_array_equal(self.problem.values, self.values)
        np.testing.assert_array_equal(self.problem.weights, self.weights)
    
    def test_representation_type(self):
        """Test that representation type is knapsack."""
        self.assertEqual(self.problem.representation_type(), "knapsack")
    
    def test_evaluate_empty_solution(self):
        """Test evaluation of empty solution."""
        solution = np.array([0, 0, 0])
        fitness = self.problem.evaluate(solution)
        self.assertEqual(fitness, 0.0)  # Negated value = 0
    
    def test_evaluate_full_solution(self):
        """Test evaluation of solution with all items."""
        solution = np.array([1, 1, 1])
        fitness = self.problem.evaluate(solution)
        # Total value = 280, but weight = 60 > 50, so infeasible
        # Penalty should make fitness worse than 0
        self.assertGreater(fitness, 0)  # Penalty applied
    
    def test_evaluate_feasible_solution(self):
        """Test evaluation of feasible solution."""
        solution = np.array([1, 1, 0])  # Items 0 and 1, weight = 30 <= 50
        fitness = self.problem.evaluate(solution)
        expected_value = 60 + 100  # = 160
        self.assertEqual(fitness, -expected_value)  # Minimization framework
    
    def test_random_solution_generation(self):
        """Test random solution generation."""
        rng = np.random.RandomState(42)
        solutions = self.problem.init_solution(rng, n=5)
        
        self.assertEqual(len(solutions), 5)
        for sol in solutions:
            self.assertEqual(len(sol), 3)
            self.assertTrue(np.all((sol == 0) | (sol == 1)))
    
    def test_clip_method(self):
        """Test that clip converts to binary."""
        X = np.array([[0.1, 0.6, 0.9], [0.3, 0.4, 0.7]])
        clipped = self.problem.clip(X)
        
        self.assertTrue(np.all((clipped == 0) | (clipped == 1)))
        np.testing.assert_array_equal(clipped, np.array([[0, 1, 1], [0, 0, 1]]))
    
    def test_greedy_repair(self):
        """Test greedy repair strategy."""
        # Infeasible solution: all items (weight = 60 > 50)
        infeasible = np.array([1, 1, 1])
        
        repaired = self.problem.greedy_repair(infeasible)
        
        # Check repaired solution is feasible
        total_weight = np.sum(repaired * self.weights)
        self.assertLessEqual(total_weight, self.capacity)
        
        # Check it's binary
        self.assertTrue(np.all((repaired == 0) | (repaired == 1)))
    
    def test_dp_solver_simple(self):
        """Test DP solver on simple instance."""
        # Known optimal: items 1 and 2 (value=220, weight=50)
        dp_value = self.problem.solve_dp()
        
        self.assertEqual(dp_value, 220)
    
    def test_dp_solver_with_seed(self):
        """Test DP solver with random instance."""
        rng = np.random.RandomState(123)
        values = rng.randint(10, 100, 10)
        weights = rng.randint(1, 50, 10)
        capacity = int(0.5 * np.sum(weights))
        
        problem = KnapsackProblem(values, weights, capacity)
        dp_value = problem.solve_dp()
        
        # Just check it runs and returns reasonable value
        self.assertGreater(dp_value, 0)
        self.assertLessEqual(dp_value, np.sum(values))
    
    def test_invalid_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        with self.assertRaises((ValueError, AssertionError)):
            KnapsackProblem(
                values=np.array([10, 20]),
                weights=np.array([5, 10, 15]),  # Different length
                capacity=100
            )
    
    def test_invalid_negative_capacity(self):
        """Test that negative capacity raises error."""
        with self.assertRaises(ValueError):
            KnapsackProblem(
                values=np.array([10, 20]),
                weights=np.array([5, 10]),
                capacity=-10
            )
    
    def test_penalty_function(self):
        """Test penalty for infeasible solutions."""
        # Infeasible solution
        infeasible = np.array([1, 1, 1])  # weight = 60 > 50
        
        # Use penalty mode
        import copy
        problem_penalty = copy.deepcopy(self.problem)
        
        fitness = problem_penalty.evaluate(infeasible)
        
        # Should have penalty (fitness > 0 despite having value)
        total_value = np.sum(infeasible * self.values)
        self.assertGreater(fitness, -total_value)  # Penalty applied


class TestKnapsackInstanceGeneration(unittest.TestCase):
    """Test Knapsack instance generation methods."""
    
    def test_uncorrelated_instance(self):
        """Test uncorrelated instance generation."""
        from benchmark.instance_generator import generate_knapsack_instance
        
        values, weights, capacity = generate_knapsack_instance(
            n_items=20,
            instance_type='uncorrelated',
            capacity_ratio=0.5,
            seed=42
        )
        
        self.assertEqual(len(values), 20)
        self.assertEqual(len(weights), 20)
        self.assertGreater(capacity, 0)
        self.assertTrue(all(values > 0))
        self.assertTrue(all(weights > 0))
    
    def test_weakly_correlated_instance(self):
        """Test weakly correlated instance generation."""
        from benchmark.instance_generator import generate_knapsack_instance
        
        values, weights, capacity = generate_knapsack_instance(
            n_items=20,
            instance_type='weakly_correlated',
            seed=42
        )
        
        # Values should be close to weights
        correlation = np.corrcoef(values, weights)[0, 1]
        self.assertGreater(correlation, 0.5)
    
    def test_strongly_correlated_instance(self):
        """Test strongly correlated instance generation."""
        from benchmark.instance_generator import generate_knapsack_instance
        
        values, weights, capacity = generate_knapsack_instance(
            n_items=20,
            instance_type='strongly_correlated',
            seed=42
        )
        
        # Values should be very close to weights + offset
        # v_i = w_i + 100
        correlation = np.corrcoef(values, weights)[0, 1]
        self.assertGreater(correlation, 0.95)
    
    def test_subset_sum_instance(self):
        """Test subset-sum instance generation."""
        from benchmark.instance_generator import generate_knapsack_instance
        
        values, weights, capacity = generate_knapsack_instance(
            n_items=20,
            instance_type='subset_sum',
            seed=42
        )
        
        # Values should equal weights
        np.testing.assert_array_equal(values, weights)


if __name__ == '__main__':
    unittest.main()