"""
Unit tests for Firefly Algorithm.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.swarm.fa import FireflyContinuousOptimizer, FireflyKnapsackOptimizer
from src.problems.continuous.rastrigin import RastriginProblem
from src.problems.discrete.knapsack import KnapsackProblem


class TestFireflyContinuousOptimizer(unittest.TestCase):
    """Test cases for continuous Firefly Algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = RastriginProblem(dim=3)
        self.optimizer = FireflyContinuousOptimizer(
            problem=self.problem,
            n_fireflies=10,
            alpha=0.2,
            beta0=1.0,
            gamma=1.0,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_fireflies, 10)
        self.assertEqual(self.optimizer.alpha, 0.2)
        self.assertEqual(self.optimizer.beta0, 1.0)
        self.assertEqual(self.optimizer.gamma, 1.0)
    
    def test_run_returns_correct_format(self):
        """Test that run() returns correct output format."""
        best_sol, best_fit, history, stats_history = self.optimizer.run(max_iter=10)
        
        self.assertEqual(len(best_sol), 3)
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 10)
        self.assertEqual(len(stats_history), 10)
        self.assertIsInstance(stats_history[0], dict)
    
    def test_convergence(self):
        """Test that algorithm converges (fitness improves)."""
        _, _, history, _ = self.optimizer.run(max_iter=50)
        
        # Final fitness should be better than or equal to initial
        self.assertLessEqual(history[-1], history[0])
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        opt1 = FireflyContinuousOptimizer(self.problem, n_fireflies=10, seed=123)
        opt2 = FireflyContinuousOptimizer(self.problem, n_fireflies=10, seed=123)
        
        _, fit1, _, _ = opt1.run(max_iter=20)
        _, fit2, _, _ = opt2.run(max_iter=20)
        
        self.assertAlmostEqual(fit1, fit2, places=10)


class TestFireflyKnapsackOptimizer(unittest.TestCase):
    """Test cases for Knapsack Firefly Algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        values = np.array([10, 20, 30, 40, 50])
        weights = np.array([1, 2, 3, 4, 5])
        capacity = 7.0
        self.problem = KnapsackProblem(values, weights, capacity)
        self.optimizer = FireflyKnapsackOptimizer(
            problem=self.problem,
            n_fireflies=10,
            alpha_flip=0.2,
            max_flips_per_move=3,
            constraint_handling="repair",
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_fireflies, 10)
        self.assertEqual(self.optimizer.alpha_flip, 0.2)
        self.assertEqual(self.optimizer.max_flips_per_move, 3)
        self.assertEqual(self.optimizer.constraint_handling, "repair")
    
    def test_run_returns_valid_solution(self):
        """Test that run() returns a valid binary solution."""
        best_sol, best_fit, history, trajectory = self.optimizer.run(max_iter=10)
        
        # Check solution validity
        self.assertEqual(len(best_sol), 5)
        self.assertTrue(np.all((best_sol == 0) | (best_sol == 1)))
        
        # Check outputs
        self.assertIsInstance(best_fit, (float, np.floating))
        self.assertEqual(len(history), 10)
        self.assertEqual(len(trajectory), 10)
    
    def test_feasibility(self):
        """Test that solutions respect capacity constraint."""
        best_sol, _, _, _ = self.optimizer.run(max_iter=20)
        
        total_weight = np.sum(best_sol * self.problem.weights)
        self.assertLessEqual(total_weight, self.problem.capacity)
    
    def test_convergence(self):
        """Test that algorithm improves solution value."""
        _, _, history, _ = self.optimizer.run(max_iter=30)
        
        # Final should be better than or equal to initial (minimization)
        self.assertLessEqual(history[-1], history[0])


if __name__ == '__main__':
    unittest.main()
