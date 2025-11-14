import random
import math
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional


class GA_solver:
    def __init__(
        self,
        filepath: Optional[str] = "TSP51.txt",
        cities: Optional[List[List]] = None,
        population_size: int = 1000,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        tournament_size: int = 5,
        elitism: int = 2,
        max_generations: int = 200,
        target: float = float("inf"),
    ):
        # Tham số GA
        self.filepath = filepath
        self.cities = cities if cities is not None else self._load_cities(filepath)
        self.N = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.max_generations = max_generations
        self.target = target

        # Lưu population dạng list of [distance, route]
        self.population: List[Tuple[float, List]] = []

    def _load_cities(self, filepath: str) -> List[List]:
        # Đọc file TSP (mỗi dòng: id x y) -> trả về list các thành phố [id, x, y]
        cities = []
        if filepath is None:
            return cities
        with open(filepath, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    cities.append([parts[0], float(parts[1]), float(parts[2])])
        return cities

    @staticmethod
    def calc_distance(route: List[List]) -> float:
        if not route:
            return 0.0
        total = 0.0
        for i in range(len(route) - 1):
            ax, ay = route[i][1], route[i][2]
            bx, by = route[i + 1][1], route[i + 1][2]
            total += math.hypot(bx - ax, by - ay)
        # cộng đoạn cuối -> đầu
        ax, ay = route[-1][1], route[-1][2]
        bx, by = route[0][1], route[0][2]
        total += math.hypot(bx - ax, by - ay)
        return total

    # Population init & evaluation
    def initialize_population(self):
        self.population = []
        for _ in range(self.N):
            route = self.cities.copy()
            random.shuffle(route)
            dist = self.calc_distance(route)
            self.population.append([dist, route])

    # Selection
    def tournament_selection(self) -> List:
        candidates = random.sample(self.population, k=self.tournament_size)
        best = min(candidates, key=lambda x: x[0])
        return best[1].copy()

    # Crossover
    def order_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        size = len(parent1)

        a, b = sorted(random.sample(range(size), 2))

        def make_child(pA, pB):
            child = [None] * size
            # copy middle segment
            child[a: b + 1] = pA[a: b + 1]
            # fill rest from pB in order
            ptr = (b + 1) % size
            for gene in pB[b+1:] + pB[:b+1]:
                if gene not in child:
                    # find next empty pos starting at ptr
                    while child[ptr] is not None:
                        ptr = (ptr + 1) % size
                    child[ptr] = gene
                    ptr = (ptr + 1) % size
            return child

        child1 = make_child(parent1, parent2)
        child2 = make_child(parent2, parent1)
        return child1, child2

    # Mutation
    def swap_mutation(self, route: List) -> List:
        # Swap mutation: chọn hai vị trí và đổi chỗ (phổ biến cho permutation).
        size = len(route)
        i, j = random.sample(range(size), 2)
        route[i], route[j] = route[j], route[i]
        return route

    # One evolution step
    def evolve_one_generation(self):
        # sắp xếp population theo fitness (distance) tăng dần (mức tốt nhất đứng đầu)
        self.population.sort(key=lambda x: x[0])
        new_pop = []

        # Elitism: giữ lại 'elitism' cá thể tốt nhất (prevents losing best solutions)
        for i in range(min(self.elitism, len(self.population))):
            # copy để tránh tham chiếu chung
            best_route = self.population[i][1].copy()
            new_pop.append([self.calc_distance(best_route), best_route])

        # Sinh phần còn lại
        while len(new_pop) < self.N:
            # Selection: tournament selection (lấy hai cha mẹ)
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover (recombination)
            if random.random() < self.crossover_rate:
                child1, child2 = self.order_crossover(parent1, parent2)
            else:
                # nếu không crossover thì sao chép cha mẹ (có thể vẫn mutate)
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self.swap_mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = self.swap_mutation(child2)

            # Thêm vào population mới
            new_pop.append([self.calc_distance(child1), child1])
            if len(new_pop) < self.N:
                new_pop.append([self.calc_distance(child2), child2])

        self.population = new_pop

    # Run GA
    def run(self):
        if not self.population:
            self.initialize_population()

        gen = 0
        best_solution = min(self.population, key=lambda x: x[0])
        best_solution_history = []
        iter_times = []

        while gen < self.max_generations:
            start_time = time.time()
            self.evolve_one_generation()
            gen += 1

            # cập nhật best
            curr_best = min(self.population, key=lambda x: x[0])
            if curr_best[0] < best_solution[0]:
                best_solution = curr_best

            best_solution_history.append(best_solution[0])

            if gen < 10 or gen % 10 == 0:
                print(f"Gen {gen}: best distance = {best_solution[0]:.6f}")

            if best_solution[0] <= self.target:
                print(f"Target reached at generation {gen}.")
                break
            end_time = time.time()
            iter_times.append(end_time - start_time)

        end_time = time.time()
        total_time = sum(iter_times)
        return best_solution[1], best_solution[0], best_solution_history, total_time, iter_times

    # Visualization
    def draw_map(self, answer: List):
        route = answer

        xs = [c[1] for c in self.cities]
        ys = [c[2] for c in self.cities]
        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, c="red")
        for city in self.cities:
            plt.annotate(city[0], (city[1], city[2]))

        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            plt.plot([a[1], b[1]], [a[2], b[2]], color="gray")

        a = route[-1]
        b = route[0]
        plt.plot([a[1], b[1]], [a[2], b[2]], color="gray")
        plt.title(f"TSP route (distance = {self.calc_distance(answer):.3f})")
        plt.show()

