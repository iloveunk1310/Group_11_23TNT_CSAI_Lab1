import numpy as np

# Rastrigin function 
# f(0) = 0
# [-5.12, 5.12]
def rastrigin(x): 
    x = np.array(x)
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=0)

# Rosenbrock function 
# f(1) = 0
# [-5, 10]
def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2, axis=0)

# Ackley function 
# f(0) = 0
# [-5, 5]
def ackley(x):
    x = np.array(x)
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=0) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=0) / n)
    return term1 + term2 + 20 + np.e

# Easom function 
# f(pi, pi) = -1
# [-100, 100]
def easom(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Easom function is only defined for 2 variables (x, y).")
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

# Beale function 
# f(3, 0.5) = 0
# [-4.5, 4.5]
def beale(x):
    x = np.array(x)
    if len(x) != 2:
        raise ValueError("Beale function is only defined for 2 variables (x, y).")
    x1, x2 = x
    return ((1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2)

# Sum of square function
# f(0, 0) = 0
# [-10, 10]
def sum_of_square(x):
    x = np.array(x)
    x1, x2 = x
    return x1**2 + x2**2

# Sphere function
# f(0, 0) = -1
# [-1, 1]
def circle(x):
    x = np.array(x)
    x1, x2 = x
    return -(50 - x1**2 - x2**2)**0.5