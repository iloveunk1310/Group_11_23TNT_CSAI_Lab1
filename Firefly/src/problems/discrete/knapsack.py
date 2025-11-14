"""
0/1 Knapsack Problem optimization.

The 0/1 Knapsack problem is a classic combinatorial optimization problem where
the goal is to select items to maximize value while staying within weight capacity.

References
----------
.. [1] https://en.wikipedia.org/wiki/Knapsack_problem
"""

import numpy as np
from typing import Literal

# Use relative import
from ...core.problem_base import ProblemBase


class KnapsackProblem(ProblemBase):
    """
    0/1 Knapsack Problem.
    
    Given a set of items, each with a weight and value, select items to maximize
    total value without exceeding the knapsack's weight capacity.
    
    Solution representation: binary vector where x[i] = 1 means item i is selected.
    
    For optimization consistency (minimization), we use:
        fitness = -total_value  (if feasible)
        fitness = -total_value + penalty_coefficient * violation  (if infeasible)
    
    This ensures:
    - Better solutions have LOWER fitness (minimize-compatible)
    - Feasible solutions always better than infeasible ones
    - Performance profiles work correctly with cost ratios
    
    Parameters
    ----------
    values : np.ndarray
        Value of each item, shape (num_items,).
    weights : np.ndarray
        Weight of each item, shape (num_items,).
    capacity : float
        Maximum weight capacity of the knapsack.
    penalty_coefficient : float, optional
        Penalty multiplier for constraint violations. Default is 1000.
    
    Attributes
    ----------
    values : np.ndarray
        Item values.
    weights : np.ndarray
        Item weights.
    capacity : float
        Knapsack capacity.
    num_items : int
        Number of items.
    penalty_coefficient : float
        Penalty for infeasible solutions.
    dp_optimal : float or None
        DP optimal value if available, None otherwise.
    
    Examples
    --------
    >>> values = np.array([10, 20, 30])
    >>> weights = np.array([1, 2, 3])
    >>> capacity = 4.0
    >>> problem = KnapsackProblem(values, weights, capacity)
    >>> x = np.array([1, 1, 0])  # Select items 0 and 1
    >>> fitness = problem.evaluate(x)
    >>> print(f"Fitness: {fitness}")  # Should be -30 (maximizing value)
    """
    
    def __init__(
        self, 
        values: np.ndarray, 
        weights: np.ndarray, 
        capacity: float,
        penalty_coefficient: float = 1000.0
    ):
        """
        Initialize Knapsack problem.
        
        Parameters
        ----------
        values : np.ndarray
            Item values, shape (n,).
        weights : np.ndarray
            Item weights, shape (n,).
        capacity : float
            Knapsack capacity (must be positive).
        penalty_coefficient : float, optional
            Penalty for constraint violation (default: 1000.0).
        
        Raises
        ------
        ValueError
            If values and weights have different lengths.
            If capacity is negative.
            If arrays are empty.
        """
        # Validate inputs
        if len(values) != len(weights):
            raise ValueError(f"values and weights must have same length. "
                            f"Got {len(values)} and {len(weights)}")
        
        if len(values) == 0:
            raise ValueError("values and weights cannot be empty")
        
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        
        self.values = np.asarray(values, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.capacity = float(capacity)
        self.num_items = len(values)
        self.penalty_coefficient = penalty_coefficient
        self.dp_optimal = None  # Will be set externally if DP is computed
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate knapsack solution with minimize-compatible fitness.
        
        Returns:
        - Feasible: fitness = -total_value (lower is better)
        - Infeasible: fitness = -total_value + penalty * violation (always worse)
        
        This ensures performance profiles work correctly:
        - Cost ratio r = fitness_algo / fitness_best always meaningful
        - Feasible solutions always have fitness < infeasible ones
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector, shape (num_items,), values in {0, 1}.
        
        Returns
        -------
        fitness : float
            Fitness value (lower is better).
        """
        # Ensure binary
        selection = (x > 0.5).astype(int)
        
        total_weight = float(np.sum(selection * self.weights))
        total_value = float(np.sum(selection * self.values))
        
        # Calculate violation
        violation = max(0.0, total_weight - self.capacity)
        
        # Base cost (minimize-compatible: negate value)
        base_cost = -total_value
        
        if violation <= 0.0:
            # Feasible solution
            fitness = base_cost
        else:
            # Infeasible: add penalty proportional to violation
            # This maintains monotonicity: more violation = worse fitness
            fitness = base_cost + self.penalty_coefficient * violation
        
        # Ensure no NaN/Inf
        if not np.isfinite(fitness):
            fitness = self.penalty_coefficient * 1e6
        
        return fitness
    
    def get_solution_info(self, x: np.ndarray) -> dict:
        """
        Get detailed information about a solution for analysis.
        
        This is used by runners/loggers to extract metrics for:
        - Performance profiles
        - Data profiles
        - Fixed-budget analysis
        - Pairwise statistical tests
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector.
        
        Returns
        -------
        info : dict
            Dictionary with keys:
            - 'Fitness': float (minimize-compatible)
            - 'Value': float (raw value, always positive)
            - 'Weight': float (total weight)
            - 'Feasible': bool (True if within capacity)
            - 'Violation': float (amount over capacity, 0 if feasible)
            - 'DP_Optimal': float or None (if available)
        """
        selection = (x > 0.5).astype(int)
        
        total_weight = float(np.sum(selection * self.weights))
        total_value = float(np.sum(selection * self.values))
        violation = max(0.0, total_weight - self.capacity)
        
        # Compute fitness (same logic as evaluate)
        base_cost = -total_value
        if violation <= 0.0:
            fitness = base_cost
        else:
            fitness = base_cost + self.penalty_coefficient * violation
        
        # Ensure no NaN/Inf
        if not np.isfinite(fitness):
            fitness = self.penalty_coefficient * 1e6
        
        return {
            'Fitness': float(fitness),
            'Value': float(total_value),
            'Weight': float(total_weight),
            'Feasible': bool(violation <= 0.0),
            'Violation': float(violation),
            'DP_Optimal': self.dp_optimal
        }
    
    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if solution is feasible (within capacity).
        
        Parameters
        ----------
        x : np.ndarray
            Binary selection vector.
        
        Returns
        -------
        feasible : bool
            True if total weight <= capacity.
        """
        selection = (x > 0.5).astype(int)
        total_weight = np.sum(selection * self.weights)
        return float(total_weight) <= self.capacity
    
    def representation_type(self) -> Literal["continuous", "tsp", "knapsack", "graph_coloring"]:
        """Return 'knapsack' for this problem type."""
        return "knapsack"
    
    def init_solution(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """
        Generate n random feasible knapsack solutions.
        
        Uses a greedy repair strategy to ensure solutions are feasible:
        randomly select items, and if capacity is exceeded, randomly remove
        items until feasible.
        
        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n : int, default=1
            Number of solutions to generate.
        
        Returns
        -------
        solutions : np.ndarray
            Array of shape (n, num_items) with binary values.
        """
        solutions = np.zeros((n, self.num_items), dtype=int)
        
        for i in range(n):
            # Start with random binary vector
            solution = rng.randint(0, 2, self.num_items)
            
            # Repair if needed
            solution = self._repair_solution(solution, rng)
            solutions[i] = solution
        
        return solutions
    
    def solve_dp(self) -> float:
        """
        Solve 0/1 Knapsack using dynamic programming.
        
        Returns
        -------
        optimal_value : float
            Optimal value achievable.
        
        Notes
        -----
        Uses standard DP with O(n*W) complexity.
        Only practical for n ≤ 200, W ≤ 100000.
        """
        # Convert to integers for DP
        int_values = self.values.astype(np.int64)
        int_weights = self.weights.astype(np.int64)
        int_capacity = int(self.capacity)
        
        n = len(int_values)
        W = int_capacity
        
        # DP table: dp[i][w] = max value using first i items with capacity w
        dp = np.zeros((n + 1, W + 1), dtype=np.int64)
        
        for i in range(1, n + 1):
            for w in range(W + 1):
                # Option 1: don't take item i-1
                dp[i][w] = dp[i-1][w]
                
                # Option 2: take item i-1 if it fits
                if int_weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], 
                                dp[i-1][w - int_weights[i-1]] + int_values[i-1])
        
        return float(dp[n][W])

    def _repair_solution(self, solution: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Repair an infeasible solution by removing items until feasible.
        
        Uses greedy strategy: removes items with lowest value/weight ratio first.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector.
        rng : np.random.RandomState
            Random number generator.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        # If already feasible, return
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        # Greedy repair: remove lowest value/weight ratio items first
        if len(selected_indices) > 0:
            # Avoid division by zero
            safe_weights = np.maximum(self.weights[selected_indices], 1e-12)
            ratios = self.values[selected_indices] / safe_weights
            sorted_indices = selected_indices[np.argsort(ratios)]  # Ascending
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def greedy_repair(self, solution: np.ndarray) -> np.ndarray:
        """
        Public method for greedy repair (for external use by optimizers).
        
        Removes items with lowest value/weight ratio first until feasible.
        
        Parameters
        ----------
        solution : np.ndarray
            Binary selection vector that may be infeasible.
        
        Returns
        -------
        repaired : np.ndarray
            Feasible binary selection vector.
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        if total_weight <= self.capacity:
            return solution
        
        selected_indices = np.where(solution == 1)[0]
        
        if len(selected_indices) > 0:
            # Calculate value/weight ratios, avoid division by zero
            safe_weights = np.maximum(self.weights[selected_indices], 1e-12)
            ratios = self.values[selected_indices] / safe_weights
            # Sort by ratio ascending (remove worst items first)
            sorted_indices = selected_indices[np.argsort(ratios)]
            
            for idx in sorted_indices:
                solution[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break
        
        return solution
    
    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        For Knapsack, clip to binary {0, 1}.
        
        Values >= 0.5 become 1, others become 0.
        
        Parameters
        ----------
        X : np.ndarray
            Solution(s), shape (n, num_items) or (num_items,).
        
        Returns
        -------
        X_binary : np.ndarray
            Binary-clipped solutions.
        """
        return (X > 0.5).astype(int)


def _hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Hamming distance between two binary vectors.
    
    Private helper for diversity calculation.
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Binary vectors.
    
    Returns
    -------
    distance : float
        Number of differing bits.
    """
    return float(np.sum(x1 != x2))


def compute_population_diversity(population: np.ndarray) -> float:
    """
    Compute population diversity for Knapsack (bitstring representation).
    
    Uses average pairwise Hamming distance, normalized by sqrt(n_items).
    
    Parameters
    ----------
    population : np.ndarray
        Population of solutions, shape (pop_size, n_items).
    
    Returns
    -------
    diversity : float
        Normalized diversity metric.
    """
    n_pop, n_items = population.shape
    
    if n_pop < 2:
        return 0.0
    
    # Sample pairs to avoid O(n^2) for large populations
    max_pairs = min(100, (n_pop * (n_pop - 1)) // 2)
    
    total_distance = 0.0
    count = 0
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for _ in range(max_pairs):
        i, j = rng.choice(n_pop, 2, replace=False)
        total_distance += _hamming_distance(population[i], population[j])
        count += 1
    
    avg_distance = total_distance / max(count, 1)
    
    # Normalize by sqrt(n_items) for scale-invariance
    diversity = avg_distance / np.sqrt(n_items)
    
    return float(diversity)

