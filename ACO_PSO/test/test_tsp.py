import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)
from src.ACO import Graph, ACO_Solver
from src.PSO import PSO_Solver
from src.GA import GA_solver


class test_tsp:
    def __init__(self):
        pass

    def load_cities(self, filepath):
        """Helper method to load cities from file and create distance matrix."""
        cities = []
        with open(filepath, "r") as f:
            for line in f:
                node_city_val = line.split()
                cities.append([float(node_city_val[1]), float(node_city_val[2])])

        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities), dtype=np.float64)

        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                city1 = cities[i]
                city2 = cities[j]
                dist = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        return cities, num_cities, distance_matrix

    def test_aco(self, filepath, alpha=0.9, beta=1.5, iters=500, ants_per_iter=10, const_q=10, const_degrade=0.9, plot=False):
        """
        Test ACO algorithm on TSP problem.
        
        """
        print("=" * 50)
        print("Testing ACO Algorithm")
        print("=" * 50)
        
        cities, num_cities, distance_matrix = self.load_cities(filepath)
        print(f"Loaded {num_cities} cities from {filepath}")
        print(f"Distance Matrix created with shape: {distance_matrix.shape}")
        
        tsp_graph = Graph(num_cities, distance_matrix)
        aco_solver = ACO_Solver(
            tsp_graph,
            alpha=alpha,
            beta=beta,
            iters=iters,
            ants_per_iter=ants_per_iter,
            const_q=const_q,
            const_degrade=const_degrade
        )
        
        solution = aco_solver.solve()
        best_route, best_length, best_lengths, iter_times = solution
        if plot:
            aco_solver.test_and_plot(tsp_graph, solution, filepath.split('.')[0].split('/')[1])
        print("\nACO Results:")
        print(f"Best route found: {best_route}")
        print(f"Best tour length: {best_length:.6f}")
        print(f"Total execution time: {sum(iter_times):.4f} seconds")
        print("=" * 50)
        
        return best_route, best_length, best_lengths, iter_times

    def test_pso(self, filepath, max_iters=200, plot = False):
        """
        Test PSO algorithm on TSP problem.
        
        """
        print("=" * 50)
        print("Testing PSO Algorithm")
        print("=" * 50)
        
        cities, num_cities, distance_matrix = self.load_cities(filepath)
        print(f"Loaded {num_cities} cities from {filepath}")
        print(f"Distance Matrix created with shape: {distance_matrix.shape}")
        
        tsp_graph = Graph(num_cities, distance_matrix)
        pso_solver = PSO_Solver(tsp_graph, max_iters=max_iters)
        solution = pso_solver.solve()
        best_route, best_length, best_lengths, iter_times = solution
        if plot:
            pso_solver.test_and_plot(solution, filepath.split('.')[0].split('/')[1])
        
        print("\nPSO Results:")
        print(f"Best route found: {best_route}")
        print(f"Best tour length: {best_length:.6f}")
        print(f"Total execution time: {sum(iter_times):.4f} seconds")
        print("=" * 50)
        
        return best_route, best_length, best_lengths, iter_times

    def test_ga(self, filepath, population_size=2000, mutation_rate=0.01, crossover_rate=0.8,
                tournament_size=5, elitism=2, max_generations=200, target=float("inf"), draw = False):
        """
        Test Genetic Algorithm on TSP problem.
        """
        print("=" * 50)
        print("Testing Genetic Algorithm")
        print("=" * 50)
        
        print(f"Loading cities from {filepath}")
        
        ga_solver = GA_solver(
            filepath=filepath,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size,
            elitism=elitism,
            max_generations=max_generations,
            target=target
        )
        
        solution = ga_solver.run()
        best_route, best_length, best_lengths, total_time, iter_times = solution
        if draw:
            ga_solver.draw_map(best_route)
        print("\nGA Results:")
        print(f"Best route found: {best_route}")
        print(f"Best tour length: {best_length:.6f}")
        print(f"Total execution time: {total_time:.4f} seconds")
        print("=" * 50)
        
        return best_route, best_length, best_lengths, total_time, iter_times

    def compare_algorithms(self, filepath, plot=False):
        """
        Compare the performance of ACO, PSO, and Genetic Algorithm on the TSP problem.
        
        """
        print("=" * 50)
        print("Comparing Algorithms")
        print("=" * 50)
        
        best_route_aco, best_len_aco, history_aco, times_aco = self.test_aco(filepath, plot=False)
        best_route_pso, best_len_pso, history_pso, times_pso = self.test_pso(filepath, plot= False)
        best_route_ga, best_len_ga, history_ga, total_time_ga, times_ga = self.test_ga(filepath, target=best_len_aco, draw=False)
        
        print("=" * 50)
        print("Algorithm Comparison Results:")
        print(f"ACO Best Route: {best_route_aco}")
        print(f"PSO Best Route: {best_route_pso}")
        print(f"GA Best Route: {best_route_ga}")
        print(f"ACO Best Length: {best_len_aco}")
        print(f"PSO Best Length: {best_len_pso}")
        print(f"GA Best Length: {best_len_ga}")
        print(f"ACO Total Time: {sum(times_aco)} seconds")
        print(f"PSO Total Time: {sum(times_pso)} seconds")
        print(f"GA Total Time: {total_time_ga} seconds")
        print("=" * 50)

        if plot:
            algorithms = ['ACO', 'PSO', 'Genetic Algorithm']
            best_lengths = [best_len_aco, best_len_pso, best_len_ga]
            name = filepath.split('.')[0].split('/')[1]
            plt.figure(figsize=(8, 6))
            plt.bar(algorithms, best_lengths, color=['blue', 'orange', 'green'])
            plt.ylabel("Best Tour Length")
            plt.title("Comparison of Best Tour Lengths by Algorithm")
            plt.grid(axis='y')
            plt.savefig("result/compare_length_"+name+".png")
            plt.show()

            # 2. 
            algorithms_time = ['ACO', 'PSO', 'Genetic Algorithm']
            total_times = [sum(times_aco), sum(times_pso), total_time_ga] # Include total_time_ga

            plt.figure(figsize=(8, 6))
            bars = plt.bar(algorithms_time, total_times, color=['blue', 'orange', 'green']) # Include green for GA
            plt.ylabel("Total Execution Time (seconds)")
            plt.title("Comparison of Total Execution Times by Algorithm") # Updated title
            plt.grid(axis='y')

            plt.savefig("result/compare_time_"+name+".png")
            plt.show()

            # 3.
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(history_aco) + 1), history_aco, label='ACO', color='blue')
            plt.plot(range(1, len(history_pso) + 1), history_pso, label='PSO', color='orange')
            plt.plot(range(1, len(history_ga) + 1), history_ga, label='GA', color='red')
            plt.xlabel("Iteration")
            plt.ylabel("Best Tour Length")
            plt.title("Best Tour Length Over Iterations (ACO vs PSO vs GA)")
            plt.legend()
            plt.grid(True)
            plt.savefig("result/compare_iterator_"+name+".png")
            plt.show()
