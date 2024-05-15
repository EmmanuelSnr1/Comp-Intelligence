import numpy as np
import random

# Define the distance matrix based on the given graph
distance_matrix = np.array([
    [0, 10, 35, 20],
    [10, 0, 35, 25],
    [35, 35, 0, 30],
    [20, 25, 30, 0]
])

# Parameters for Firefly Algorithm
num_fireflies = 10
num_iterations = 100
alpha = 0.5  # Initial randomness factor
beta0 = 1    # Initial attractiveness at distance 0
gamma = 1    # Light absorption coefficient

# Initialize population of fireflies (random tours)
def init_fireflies(num_fireflies, num_cities):
    fireflies = []
    for _ in range(num_fireflies):
        tour = np.random.permutation(num_cities)
        fireflies.append(tour)
    return fireflies

# Calculate tour cost
def calculate_cost(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    cost += distance_matrix[tour[-1], tour[0]]  # Return to start
    return cost

# Move firefly i towards firefly j
def move_firefly(firefly_i, firefly_j, alpha, beta, distance_matrix):
    new_firefly = firefly_i.copy()
    for k in range(len(firefly_i)):
        if random.random() < beta:
            new_firefly[k] = firefly_j[k]
    if random.random() < alpha:
        swap_indices = np.random.choice(len(firefly_i), 2, replace=False)
        new_firefly[swap_indices[0]], new_firefly[swap_indices[1]] = new_firefly[swap_indices[1]], new_firefly[swap_indices[0]]
    return new_firefly

# Perform Levy flight
def levy_flight(Lambda):
    sigma1 = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
              (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=None)
    v = np.random.normal(0, sigma2, size=None)
    step = u / abs(v) ** (1 / Lambda)
    return step

# Exploration-enhanced Firefly Algorithm
def firefly_algorithm_exploration(distance_matrix, num_fireflies, num_iterations, alpha, beta0, gamma):
    num_cities = distance_matrix.shape[0]
    fireflies = init_fireflies(num_fireflies, num_cities)
    best_tour = None
    best_cost = float('inf')

    for iteration in range(num_iterations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if i != j:
                    cost_i = calculate_cost(fireflies[i], distance_matrix)
                    cost_j = calculate_cost(fireflies[j], distance_matrix)
                    if cost_j < cost_i:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = beta0 * np.exp(-gamma * r ** 2)
                        fireflies[i] = move_firefly(fireflies[i], fireflies[j], alpha, beta, distance_matrix)
                        cost_i = calculate_cost(fireflies[i], distance_matrix)
                        if cost_i < best_cost:
                            best_cost = cost_i
                            best_tour = fireflies[i]

        # Increase randomness (alpha) over time to encourage exploration
        alpha = 0.5 * (1 - iteration / num_iterations)

        # Random walks for exploration
        for i in range(num_fireflies):
            if random.random() < 0.1:  # 10% chance to perform a random walk
                fireflies[i] = np.random.permutation(num_cities)

        # Detect stagnation and reinitialize part of the population if needed
        if iteration > 0 and iteration % 20 == 0:
            # Reinitialize 20% of the population
            num_reinit = int(0.2 * num_fireflies)
            for _ in range(num_reinit):
                fireflies[random.randint(0, num_fireflies - 1)] = np.random.permutation(num_cities)

    return best_tour, best_cost

# Run the Exploration-focused Firefly Algorithm
best_tour, best_cost = firefly_algorithm_exploration(distance_matrix, num_fireflies, num_iterations, alpha, beta0, gamma)
print("Best tour:", best_tour)
print("Best cost:", best_cost)
