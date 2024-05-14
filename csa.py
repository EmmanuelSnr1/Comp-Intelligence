import numpy as np
import random

# Define the distance matrix based on the given graph
distance_matrix = np.array([
    [0, 10, 35, 20],
    [10, 0, 35, 25],
    [35, 35, 0, 30],
    [20, 25, 30, 0]
])

# Parameters for Cuckoo Search Algorithm
num_nests = 10
num_iterations = 100
pa = 0.5  # Increased probability of abandonment for exploitation

# Initialize population of nests (random tours)
def init_nests(num_nests, num_cities):
    nests = []
    for _ in range(num_nests):
        tour = np.random.permutation(num_cities)
        nests.append(tour)
    return nests

# Calculate tour cost
def calculate_cost(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    cost += distance_matrix[tour[-1], tour[0]]  # Return to start
    return cost

# Perform Levy flight
def levy_flight(Lambda):
    # Reduced step size for more local exploration
    sigma1 = ((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)) / 10
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=None)
    v = np.random.normal(0, sigma2, size=None)
    step = u / abs(v) ** (1 / Lambda)
    return step

# Generate new solution via Levy flight
def generate_new_solution(tour):
    new_tour = tour.copy()
    step = int(abs(levy_flight(1.5))) % len(tour)
    swap_indices = np.random.choice(len(tour), 2, replace=False)
    new_tour[swap_indices[0]], new_tour[swap_indices[1]] = new_tour[swap_indices[1]], new_tour[swap_indices[0]]
    return new_tour

# Exploitation-focused Cuckoo Search Algorithm
def cuckoo_search_exploitation(distance_matrix, num_nests, num_iterations, pa):
    num_cities = distance_matrix.shape[0]
    nests = init_nests(num_nests, num_cities)
    best_tour = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        new_nests = []
        for nest in nests:
            new_tour = generate_new_solution(nest)
            new_cost = calculate_cost(new_tour, distance_matrix)
            old_cost = calculate_cost(nest, distance_matrix)
            if new_cost < old_cost:
                new_nests.append(new_tour)
            else:
                new_nests.append(nest)
        
        # Sort nests by their quality
        nests = sorted(new_nests, key=lambda x: calculate_cost(x, distance_matrix))
        
        # Apply an aggressive replacement strategy
        num_replace = int(pa * num_nests)
        for i in range(num_replace):
            if random.random() < pa:  # Only replace with better solutions
                new_tour = generate_new_solution(nests[i])
                if calculate_cost(new_tour, distance_matrix) < calculate_cost(nests[i], distance_matrix):
                    nests[i] = new_tour

        # Retain the best solution unchanged
        current_best_tour = nests[0]
        current_best_cost = calculate_cost(current_best_tour, distance_matrix)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_tour = current_best_tour

    return best_tour, best_cost

# Run the Exploitation-focused Cuckoo Search Algorithm
best_tour, best_cost = cuckoo_search_exploitation(distance_matrix, num_nests, num_iterations, pa)
print("Best tour:", best_tour)
print("Best cost:", best_cost)
