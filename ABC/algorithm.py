import random
import numpy as np

def hill_climbing(function, max_iterations=500, step_size=0.1, x_range=(-5, 5), y_range=(-5, 5)):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    current_value = function([x, y])
    history = [(x, y, current_value)]

    for i in range(max_iterations):
        x_new = x + random.uniform(-step_size, step_size)
        y_new = y + random.uniform(-step_size, step_size)
        x_new = np.clip(x_new, *x_range)
        y_new = np.clip(y_new, *y_range)

        new_value = function([x_new, y_new])

        if new_value < current_value:
            x, y, current_value = x_new, y_new, new_value

        if (i + 1) % 10 == 0:
            history.append((x, y, current_value))
    return x, y, current_value, history


def simulated_annealing(function, max_iterations=1000, x_range=(-5, 5), y_range=(-5, 5), initial_temp=10.0, cooling_rate=0.99):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    current_value = function([x, y])
    
    x_best, y_best, z_best = x, y, current_value
    
    T = initial_temp  # Nhiệt độ ban đầu
    history = [(x, y, current_value)]
    
    for i in range(max_iterations):
        # Sinh lân cận ngẫu nhiên
        x_new = x + random.uniform(-0.1, 0.1)
        y_new = y + random.uniform(-0.1, 0.1)
        
        # Giới hạn trong khoảng cho phép
        x_new = np.clip(x_new, *x_range)
        y_new = np.clip(y_new, *y_range)
        
        # Tính giá trị mới
        new_value = function([x_new, y_new])
        delta = new_value - current_value
        
        # Nếu tốt hơn, chấp nhận
        # Nếu xấu hơn, chấp nhận với xác suất exp(-delta / T)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x, y, current_value = x_new, y_new, new_value
        
        # Cập nhật nghiệm tốt nhất
        if current_value < z_best:
            x_best, y_best, z_best = x, y, current_value
        
        # Giảm nhiệt độ
        T *= cooling_rate
        
        # lưu history
        if (i + 1) % 50 == 0:
            history.append((x, y, current_value))
    
    return x_best, y_best, z_best, history


def genetic_algorithm(function, population_size=50, generations=200,
                      x_range=(-5, 5), y_range=(-5, 5),
                      crossover_rate=0.8, mutation_rate=0.1):

    population = np.random.uniform(
        low=[x_range[0], y_range[0]],
        high=[x_range[1], y_range[1]],
        size=(population_size, 2)
    )

    fitness = np.array([function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    min_fit = fitness[best_idx]
    
    history = []
    
    for gen in range(generations):
        shifted = fitness - fitness.min() + 1e-8
        inv_fit = 1.0 / (shifted + 1e-8) 
        probs = inv_fit / inv_fit.sum()
        
        selected_indices = np.random.choice(
            np.arange(population_size),
            size=population_size,
            replace=True,
            p=probs
        )
        mating_pool = population[selected_indices]
        
        # Lai ghép
        offspring = []
        for i in range(0, population_size, 2):
            p1 = mating_pool[i]
            p2 = mating_pool[(i + 1) % population_size]
            
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            offspring.extend([c1, c2])
        
        offspring = np.array(offspring[:population_size])
        
        # Đột biến
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation = np.random.uniform(-0.5, 0.5, 2)
                offspring[i] += mutation
                offspring[i, 0] = np.clip(offspring[i, 0], *x_range)
                offspring[i, 1] = np.clip(offspring[i, 1], *y_range)
        
        # Cập nhật 
        population = offspring
        fitness = np.array([function(ind) for ind in population])
        
        # Cập nhật best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < min_fit:
            min_fit = fitness[current_best_idx]
            best = population[current_best_idx].copy()
        
        # Lưu history
        if gen < 10 or (gen + 1) % 10 == 0 or gen == generations - 1:
            history.append(population.copy())
    
    return best[0], best[1], min_fit, history


def abc(function, SN=20, limit=50, maxCycle=500, x_range=(-5, 5), y_range=(-5, 5)):
    D = 2
    xmin, xmax = x_range
    ymin, ymax = y_range

    # Khởi tạo quần thể (SN con ong)
    food = np.random.uniform([xmin, ymin], [xmax, ymax], (SN, D))
    fitness = np.array([function(food[i]) for i in range(SN)])
    trial = np.zeros(SN)

    # Nghiệm tốt nhất
    best_idx = np.argmin(fitness)
    best = food[best_idx].copy()
    best_fit = fitness[best_idx]

    history = []  # lưu vị trí các ong qua từng giai đoạn

    for cycle in range(maxCycle):

        # Pha ong thợ
        for i in range(SN):
            k = np.random.randint(0, SN)
            while k == i:
                k = np.random.randint(0, SN)

            phi = np.random.uniform(-1, 1, D)
            v = food[i] + phi * (food[i] - food[k])
            v[0] = np.clip(v[0], xmin, xmax)
            v[1] = np.clip(v[1], ymin, ymax)

            f_new = function(v)
            if f_new < fitness[i]:
                food[i] = v
                fitness[i] = f_new
                trial[i] = 0
            else:
                trial[i] += 1

        # Pha ong quan sát
        prob = (1 / (1 + fitness)) / np.sum(1 / (1 + fitness))
        for i in range(SN):
            if np.random.rand() < prob[i]:
                k = np.random.randint(0, SN)
                while k == i:
                    k = np.random.randint(0, SN)
                phi = np.random.uniform(-1, 1, D)
                v = food[i] + phi * (food[i] - food[k])
                v[0] = np.clip(v[0], xmin, xmax)
                v[1] = np.clip(v[1], ymin, ymax)

                f_new = function(v)
                if f_new < fitness[i]:
                    food[i] = v
                    fitness[i] = f_new
                    trial[i] = 0
                else:
                    trial[i] += 1

        # Pha ong do thám
        for i in range(SN):
            if trial[i] > limit:
                food[i] = np.random.uniform([xmin, ymin], [xmax, ymax])
                fitness[i] = function(food[i])
                trial[i] = 0

        # Cập nhật nghiệm tốt nhất
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best_fit = fitness[current_best_idx]
            best = food[current_best_idx].copy()

        # lưu history
        if (cycle + 1) % 10 == 0:
            positions = [(food[i, 0], food[i, 1], fitness[i]) for i in range(SN)]
            history.append(positions)

    return best[0], best[1], best_fit, history

