import numpy as np
import random
import time
import matplotlib.pyplot as plt


class Graph():
    def __init__(self, count_nodes, dist, phero_lv=None):
        self.count_nodes = count_nodes
        self.dist = dist
        if phero_lv:
            self.phero = np.full_like(dist, phero_lv).astype('float64')
        else:
            self.phero = np.full_like(dist, self.dist.mean() * 10).astype('float64')

    def __str__(self):
        return f'node: {str(self.count_nodes)}, distance: {self.dist}, phero_weight: {self.phero}'


class ACO_Solver:
    def __init__(self, graph, alpha=0.9, beta=1.5, iters=10, ants_per_iter=10, const_q=10, const_degrade=0.9):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.ants_per_iter = ants_per_iter
        self.const_q = const_q
        self.const_degrade = const_degrade

    def _len_cycle(self, cycle):
        length = 0
        i = 0
        while i < len(cycle) - 1:
            length += self.graph.dist[cycle[i]][cycle[i+1]]
            i += 1
        length += self.graph.dist[cycle[i]][cycle[0]]
        return length

    def _travel_graph(self, start_node=0):
        cycle = [start_node]
        count = self.graph.count_nodes
        pass_node = np.asarray([0 for i in range(count)])
        pass_node[start_node] = 1
        curr_node = start_node
        total = 0
        while len(cycle) < count:
            edge_val = []
            near_node = []
            for node in range(count):
                if pass_node[node] == 0:
                    phero_lv = max(self.graph.phero[curr_node][node], 1e-9)
                    weight = (phero_lv**self.alpha) / (self.graph.dist[curr_node][node]**self.beta)
                    edge_val.append(weight)
                    near_node.append(node)

            if not near_node:
                break

            next_node = random.choices(near_node, weights=edge_val)[0]
            pass_node[next_node] = 1
            cycle.append(next_node)
            curr_node = next_node

        total_len = self._len_cycle(cycle)
        return cycle, total_len

    def solve(self):
        curr_best = None
        curr_best_len = float("inf")
        old_best = None
        limit = 0
        patience = 100
        max_patience = 150  # thêm max_patience để early stopping

        best_lengths = []
        iter_times = []

        for iter in range(self.iters):
            start_time = time.time()
            if (iter < 10 or iter % 10 == 0):
                print('-' * 30)
                print("Iteration: ", iter + 1, "/", self.iters)

            cycle_ants = []
            for i in range(self.ants_per_iter):
                res = self._travel_graph(random.randint(0, self.graph.count_nodes - 1))
                cycle_ants.append(res)

            cycle_ants.sort(key=lambda x: x[1])

            # Lưu giá trị tốt nhất hiện tại
            old_best_len_this_iter = curr_best_len

            if curr_best:
                cycle_ants.append((curr_best, curr_best_len))
                old_best = curr_best_len

            for cycle, total_len in cycle_ants:
                if total_len < curr_best_len:
                    curr_best = cycle
                    curr_best_len = total_len

                plus = self.const_q/total_len
                i = 0
                while i < len(cycle) - 1:
                    self.graph.phero[cycle[i]][cycle[i+1]] += plus
                    i += 1
                self.graph.phero[cycle[i]][cycle[0]] += plus
                self.graph.phero *= self.const_degrade

            # Kiểm tra sự phát triển với giới hạn (nếu giữ nguyên liên tục)
            if curr_best_len < old_best_len_this_iter:
                limit = 0  
            else:
                limit += 1  

            if limit >= patience:
                self.graph.phero += self.graph.phero.mean()
            if limit >= max_patience:
                print("Early stopping due to no improvement in", max_patience, "iterations.")
                break

            end_time = time.time()
            best_lengths.append(curr_best_len)
            iter_times.append(end_time - start_time)
            if (iter < 10 or iter % 10 == 0):
                print("Best current solution: ", curr_best_len)

        return curr_best, curr_best_len, best_lengths, iter_times

    def test_and_plot(self, graph, solve_method=None, name = ""):
        if solve_method:
            solution, best_len, best_lengths, iter_times = solve_method
        else:
            solution, best_len, best_lengths, iter_times = self.solve()
        print("Best solution:", solution)
        print("Best length:", best_len)

        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(best_lengths) + 1), best_lengths)
        plt.xlabel("Iteration")
        plt.ylabel("Best Length")
        plt.title("Best Tour Length After Each Iteration (ACO)")
        plt.grid(True)
        plt.savefig("result/ACO_" + name +".png")
        plt.show()

        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(iter_times) + 1), iter_times)
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title("Time Taken for Each Iteration")
        plt.grid(True)
        plt.show()


