import algorithm as _a
import function as _f
import graph as _g
import time
import tracemalloc

def stimulate(algorithm, function):
    x_best, y_best, val_best, history = algorithm(function)
    if algorithm == _a.abc:
        _g.animate_abc(history, function) 
    elif algorithm == _a.genetic_algorithm:
        _g.animate_genetic(history, function)
    else:
        _g.animate_optimization(history, function)
    print(f"Giá trị nhỏ nhất của hàm số là: {val_best}, đạt được tại (x, y) = ({x_best}, {y_best})")

def measure_performance(algorithm, function, x_expected, y_expected, val_expected):
    tracemalloc.start()               # Bắt đầu đo bộ nhớ
    start_time = time.time()          # Bắt đầu đo thời gian
    
    x_best, y_best, val_best, history = algorithm(function)
    
    end_time = time.time()            # Kết thúc đo thời gian
    current, peak = tracemalloc.get_traced_memory()  # Lấy bộ nhớ hiện tại và cực đại
    tracemalloc.stop()
    
    print(f"Thuật toán: {algorithm.__name__}")
    print(f"Thời gian chạy: {end_time - start_time:.6f} giây")
    print(f"Bộ nhớ dùng (peak): {peak / 1024:.2f} KB")
    
    if (x_best - x_expected) < 1.0e-6 and (y_best - y_expected) < 1.0e-6 and (val_best - val_expected) < 1.0e-6:
        print("Kết quả đúng\n")
    else:
        print(f"Kết quả sai hoặc sai số lớn: Giá trị nhỏ nhất là {val_best} tại điểm [{x_best}, {y_best}].\nKỳ vọng: Giá trị nhỏ nhất là {val_expected} tại điểm [{x_expected}, {y_expected}].\n")
    return


# Trực quan hoá thuật toán
# stimulate(_a.abc, _f.rosenbrock)


# Đo hiệu năng

# Rastrigin function, f(0) = 0, [-5.12, 5.12]

# Rosenbrock function, f(1) = 0, [-5, 10]
algo_list = [
    _a.abc,
    _a.hill_climbing,
    _a.simulated_annealing,
    _a.genetic_algorithm
]
for algo in algo_list:
    measure_performance(algo, _f.circle, 1, 1, 0)