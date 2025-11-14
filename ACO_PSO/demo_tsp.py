import sys
import os

# Add current directory to path to import from test
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

from test.test_tsp import test_tsp

tester = test_tsp()

# Test ACO
best_route_aco, best_len_aco, history_aco, times_aco = tester.test_aco("data/small.txt", plot=True)

# Test PSO
best_route_pso, best_len_pso, history_pso, times_pso = tester.test_pso("data/small.txt", plot = True)

# Test GA
best_route_ga, best_len_ga, history_ga, total_time_ga, times_ga = tester.test_ga("data/small.txt", target=best_len_aco, draw=True)

tester.compare_algorithms("data/small.txt", True)