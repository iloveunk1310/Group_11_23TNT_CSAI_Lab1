import numpy as np
import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


class Agent:
    def __init__(self, agent_id: int = 0, location_count: int = 0):
        self.agent_id = agent_id
        self.location_count = location_count

        # Khởi tạo hành trình (bắt đầu và kết thúc tại 0)
        permuted = np.random.permutation(np.arange(1, location_count + 1))
        self.route = np.array([0, *permuted.tolist(), 0], dtype=int)

        # swaps là danh sách các cặp hoán đổi (i, j) hoặc dấu hiệu 'RESHUFFLE'
        self.swaps: Union[List[Tuple[int, int]], str] = []

        # Lưu trữ cá nhân tốt nhất
        self.personal_best_route = self.route.copy()
        self.personal_best_score = float('inf')

    def compute_score(self, distance_matrix: np.ndarray) -> float:
        # Tính tổng khoảng cách của route hiện tại bằng zip trên cặp điểm liên tiếp.
        # Nếu tốt hơn personal best thì cập nhật.
        total = 0.0
        # zip qua các cặp liên tiếp (prev, curr)
        for prev_loc, curr_loc in zip(self.route[:-1], self.route[1:]):
            total += float(distance_matrix[prev_loc, curr_loc])

        # Cập nhật personal best nếu cần
        if total < self.personal_best_score:
            self.personal_best_score = total
            self.personal_best_route = self.route.copy()

        return total

    def compute_swaps(self,
                      global_best_score: float,
                      global_best_route: np.ndarray,
                      w: float = 0.3,
                      c1: float = 0.6,
                      c2: float = 1.0):
        # Tạo danh sách hoán đổi mới dựa trên ảnh hưởng cá nhân và toàn cục.
        # Nếu chưa có best toàn cục hoặc muốn xáo trộn ngẫu nhiên -> đặt cờ
        if global_best_score == float('inf') or random.random() > 0.85:
            self.swaps = 'RESHUFFLE'
            return

        proposed_swaps: List[Tuple[int, int]] = []
        used_positions = set()

        # Duyệt các chỉ số của địa điểm trong route (bỏ vị trí 0 ở hai đầu)
        indices = list(range(1, self.location_count + 1))
        # Thay đổi thứ tự xử lý để khác so với vòng for cố định
        random.shuffle(indices)

        for pos in indices:
            # inertia: giữ nguyên (không hoán đổi) với xác suất inertia
            if random.random() < w:
                used_positions.add(pos)
                continue

            r = random.random()
            # ưu tiên ảnh hưởng cá nhân (cognitive), sau đó ảnh hưởng toàn cục (social)
            if r < c1:
                # tìm vị trí của location hiện tại trong personal best route
                loc = self.route[pos]
                found = np.flatnonzero(self.personal_best_route == loc)
                if found.size == 0:
                    continue
                swap_pos = int(found[0])
                if pos not in used_positions and swap_pos not in used_positions:
                    proposed_swaps.append((pos, swap_pos))
                    used_positions.update({pos, swap_pos})

            elif r < c1 + c2:
                # tìm vị trí của location hiện tại trong global best route
                loc = self.route[pos]
                found = np.flatnonzero(global_best_route == loc)
                if found.size == 0:
                    continue
                swap_pos = int(found[0])
                if pos not in used_positions and swap_pos not in used_positions:
                    proposed_swaps.append((pos, swap_pos))
                    used_positions.update({pos, swap_pos})

            # else: không làm gì (bỏ qua), tiếp vị trí tiếp theo

        self.swaps = proposed_swaps

    def apply_swaps(self):
        if self.swaps == 'RESHUFFLE':
            # lấy hai chỉ số ngẫu nhiên (không lấy điểm 0 ở hai đầu)
            i, j = random.sample(range(1, self.location_count + 1), 2)
            self.route[i], self.route[j] = self.route[j], self.route[i]
        else:
            # thực hiện từng hoán đổi (i, j)
            for a, b in self.swaps:
                # đảm bảo là indices hợp lệ trong mảng route
                if 0 <= a < len(self.route) and 0 <= b < len(self.route):
                    self.route[a], self.route[b] = self.route[b], self.route[a]


class PSO_Solver:
    def __init__(self, graph, max_iters: int = 300):
        self.graph = graph
        self.max_iters = max_iters
        # số địa điểm (không tính node 0 là start/end)
        self.num_locations = graph.count_nodes - 1
        # số particle dựa trên số node như trước
        self.num_particles = ((4 * (graph.count_nodes - 1)) ** 2)

    def solve(self):
        # Ma trận khoảng cách
        distance_matrix = self.graph.dist

        print(f"Number of agents (particles): {self.num_particles}")
        # Khởi tạo agents
        agents = [Agent(agent_id + 1, self.num_locations) for agent_id in range(self.num_particles)]

        epsilon = []
        # theo dõi best toàn cục
        global_best_route = None
        global_best_score = float('inf')

        best_score_history = []
        iter_times = []

        for iter in range(self.max_iters):
            start_time = time.time()

            # Đánh giá fitness / score cho từng agent
            for ag in agents:
                score = ag.compute_score(distance_matrix)
                # compute_score đã cập nhật personal best bên trong
                # cập nhật global best nếu cần
                if score < global_best_score:
                    global_best_score = score
                    global_best_route = ag.route.copy()

            epsilon.append(global_best_score)

            # Cập nhật "vận tốc" (swaps) và áp dụng chúng
            for ag in agents:
                ag.compute_swaps(global_best_score, global_best_route)
                ag.apply_swaps()

            end_time = time.time()
            best_score_history.append(global_best_score)
            iter_times.append(end_time - start_time)

            if (iter < 10 or iter % 10 == 0):
                print('-' * 30)
                print(f'Iteration: {iter + 1}')
                print(f'global_best_score: {global_best_score}')

            # Kiểm tra hội tụ (early stopping)
            if len(epsilon) >= 50:
                if len(set(epsilon[-25:])) == 1:
                    print(f"Early stopping at iteration {iter + 1} due to convergence.")
                    break

        return global_best_route, global_best_score, best_score_history, iter_times

    def test_and_plot(self, solve_method = None, name = ""):
        if solve_method:
            solution, best_len, best_lengths, iter_times = solve_method
        else:
            solution, best_len, best_lengths, iter_times = self.solve()
        print("Best route:", solution)
        print("Best length:", best_len)

        # Plot best length sau mỗi iteration
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(best_lengths) + 1), best_lengths)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Travelled Distance)")
        plt.title("Best Tour length After Each Iteration (Refactored PSO)")
        plt.grid(True)
        plt.savefig("result/PSO_" + name +".png")
        plt.show()

        # Plot thời gian mỗi iteration
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(iter_times) + 1), iter_times)
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title("Time Taken for Each Iteration (Refactored PSO)")
        plt.grid(True)
        plt.show()
        

