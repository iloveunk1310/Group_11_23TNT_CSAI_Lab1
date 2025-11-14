"""
Generate Knapsack problem instances for benchmarking.
Supports 4 instance types: uncorrelated, weakly, strongly, inverse.
"""

import numpy as np
from typing import Tuple


def generate_knapsack_instance(
    n_items: int,
    instance_type: str,
    seed: int,
    capacity_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a Knapsack instance.
    
    Parameters
    ----------
    n_items : int
        Number of items.
    instance_type : str
        Type of instance: 'uncorrelated', 'weakly'/'weakly_correlated', 
        'strongly'/'strongly_correlated', 'inverse'/'inverse_strongly'.
    seed : int
        Random seed for reproducibility.
    capacity_ratio : float, optional
        Fraction of total weight to use as capacity (default: 0.5).
    
    Returns
    -------
    values : np.ndarray
        Item values, shape (n_items,), dtype=int.
    weights : np.ndarray
        Item weights, shape (n_items,), dtype=int.
    capacity : int
        Knapsack capacity, dtype=int.
    """
    rng = np.random.RandomState(seed)
    
    # Normalize instance type names (accept both short and long forms)
    type_map = {
        'weakly_correlated': 'weakly',
        'strongly_correlated': 'strongly',
        'inverse_strongly': 'inverse',
        'subset_sum': 'subset'  # Keep for backward compatibility but not used
    }
    normalized_type = type_map.get(instance_type, instance_type)
    
    if normalized_type == 'uncorrelated':
        values = rng.randint(1, 1001, n_items)
        weights = rng.randint(1, 1001, n_items)
    
    elif normalized_type == 'weakly':
        weights = rng.randint(1, 1001, n_items)
        values = weights + rng.randint(-100, 101, n_items)
        values = np.maximum(values, 1)
    
    elif normalized_type == 'strongly':
        weights = rng.randint(1, 1001, n_items)
        values = weights + 100
    
    elif normalized_type == 'inverse':
        # Inverse strongly correlated: heavier items have lower value
        # This creates a different optimization landscape than strongly correlated
        weights = rng.randint(1, 1001, n_items)
        # Inverse relationship: v_i = (max_weight + 100) - w_i
        max_weight = np.max(weights)
        values = (max_weight + 100) - weights
        values = np.maximum(values, 1)  # Ensure all values are positive
    
    elif normalized_type == 'subset':
        # Keep subset-sum for backward compatibility but not recommended
        weights = rng.randint(1, 1001, n_items)
        values = weights.copy()
    
    else:
        valid_types = ['uncorrelated', 'weakly', 'weakly_correlated', 
                       'strongly', 'strongly_correlated', 'inverse', 'inverse_strongly',
                       'subset', 'subset_sum']
        raise ValueError(f"Unknown instance_type: {instance_type}. Must be one of {valid_types}")
    
    # Ensure integer types for DP compatibility
    values = values.astype(np.int64)
    weights = weights.astype(np.int64)
    capacity = int(capacity_ratio * np.sum(weights))
    
    return values, weights, capacity


if __name__ == "__main__":
    # Test instance generation
    print("Testing Knapsack Instance Generation")
    print("=" * 60)
    
    for inst_type in ['uncorrelated', 'weakly', 'strongly', 'inverse']:
        values, weights, capacity = generate_knapsack_instance(50, inst_type, 42)
        print(f"\n{inst_type.upper()}:")
        print(f"  Items: {len(values)}")
        print(f"  Capacity: {capacity}")
        print(f"  Total weight: {np.sum(weights)}")
        print(f"  Value range: [{np.min(values)}, {np.max(values)}]")
        print(f"  Weight range: [{np.min(weights)}, {np.max(weights)}]")
        
        if inst_type == 'inverse':
            # For inverse, check that heavier items generally have lower value
            correlation = np.corrcoef(weights, values)[0, 1]
            print(f"  Weight-Value correlation: {correlation:.3f} (should be negative)")
        elif inst_type == 'strongly':
            # For strongly correlated, check positive correlation
            correlation = np.corrcoef(weights, values)[0, 1]
            print(f"  Weight-Value correlation: {correlation:.3f} (should be positive)")