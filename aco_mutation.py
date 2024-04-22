import numpy as np
import random
import logging
import time

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Problem definition
# stock_lengths = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
# stock_costs = [86, 85, 83, 79, 68, 66, 64, 63]
# piece_lengths = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# quantities = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

# Problem Definition
stock_lengths = [120, 115, 110, 105, 100]
stock_costs = [12, 11.5, 11, 10.5, 10]
piece_lengths = [21, 22, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66, 67]
quantities = [13, 15, 7, 5, 9, 9, 3, 15, 18, 17, 4, 17, 20, 9, 4, 19, 4, 12, 15, 3, 20, 14, 15, 6, 4, 7, 5, 19, 19, 6, 3, 7, 20, 5, 10, 17]
2e


# ACO Parameters
num_ants = 10
num_iterations = 100
alpha = 1.0
beta = 1.0
decay = 0.1
initial_pheromone = 0.1
mutation_base_rate = 0.01

# Initialize pheromones
pheromones = np.full((len(stock_lengths), len(piece_lengths)), initial_pheromone)

def heuristic_value(stock_index, piece_index):
    piece_utilization = piece_lengths[piece_index] / stock_lengths[stock_index]
    return piece_utilization / stock_costs[stock_index]

def mutate(solution, mutation_rate):
    for stock_index, activities in enumerate(solution):
        if random.random() < mutation_rate:
            for activity in activities:
                if activity and random.random() < mutation_rate:
                    piece_to_mutate = random.choice(activity)
                    activity.remove(piece_to_mutate)
                    # Try to insert the mutated piece into a different position
                    possible_positions = range(len(piece_lengths))
                    new_position = random.choice(possible_positions)
                    activity.insert(new_position, piece_to_mutate)
    logging.debug("Post-mutation solution: {}".format(solution))

def adjust_mutation_rate(previous_cost, current_cost, base_rate):
    if current_cost < previous_cost:  # Improvement found
        return max(base_rate / 2, 0.001)  # Decrease mutation rate
    else:
        return min(base_rate * 2, 0.1)  # Increase mutation rate if stagnated

def solve_aco():
    best_solution = None
    best_cost = float('inf')
    current_mutation_rate = mutation_base_rate

    for iteration in range(num_iterations):
        solutions = []
        for _ in range(num_ants):
            remaining_quantities = quantities[:]
            solution = construct_solution(pheromones, remaining_quantities)
            cost = calculate_fitness(solution)
            solutions.append((solution, cost))
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
            logging.debug(f"Iteration {iteration}, Ant {_}, Cost: {cost}, Solution: {solution}")

        # Update pheromones
        update_pheromones(pheromones, solutions, best_cost)

        # Mutation step
        for solution, cost in solutions:
            mutate(solution, current_mutation_rate)

        # Adjust mutation rate based on performance
        current_mutation_rate = adjust_mutation_rate(best_cost, cost, current_mutation_rate)
        logging.debug(f"Current mutation rate: {current_mutation_rate}")

    return best_solution, best_cost

def construct_solution(pheromones, remaining_quantities):
    solution = []
    remaining_quantities = remaining_quantities[:]
    for stock_index in range(len(stock_lengths)):
        activities = []
        while any(remaining_quantities):
            activity = []
            current_length = stock_lengths[stock_index]
            while current_length > 0 and any(remaining_quantities):
                probs = [pheromones[stock_index][j] * alpha * heuristic_value(stock_index, j) * beta
                         if remaining_quantities[j] > 0 and piece_lengths[j] <= current_length else 0
                         for j in range(len(piece_lengths))]
                total_prob = sum(probs)
                if total_prob > 0:
                    probs /= total_prob
                    chosen_piece_index = np.random.choice(len(piece_lengths), p=probs)
                    activity.append(chosen_piece_index)
                    remaining_quantities[chosen_piece_index] -= 1
                    current_length -= piece_lengths[chosen_piece_index]
                else:
                    break
            if activity:
                activities.append(activity)
        if activities:
            solution.append((stock_index, activities))
    return solution

def update_pheromones(pheromones, solutions, best_cost):
    for solution, cost in solutions:
        for stock_index, activities in solution:
            for activity in activities:
                for piece_index in activity:
                    pheromones[stock_index][piece_index] += 1 / (cost + 1)
    pheromones *= (1 - decay)

def calculate_fitness(solution):
    cost = 0
    for stock_index, activities in solution:
        for activity in activities:
            if activity:
                cost += stock_costs[stock_index]
    return cost

def calculate_waste(solution):
    total_waste = 0
    for stock_index, activities in solution:
        for activity in activities:
            used_length = sum(piece_lengths[piece_index] for piece_index in activity)
            waste_per_piece = stock_lengths[stock_index] - used_length
            total_waste += waste_per_piece
    return total_waste


def print_solution(solution, cost):
    start_time = time.time()
    best_solution, best_cost = solve_aco()
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Best Cost: {cost}")
    print("Total waste :", calculate_waste (solution))
    print("Computation time :", computation_time)
    print("Solution:")
    for stock_index, activities in solution:
        for activity in activities:
            pieces = [piece_lengths[piece_index] for piece_index in activity]
            print(f"Stock Type {stock_index} (Length {stock_lengths[stock_index]}): Pieces cut: {pieces}")

# Main execution block
if __name__ == "__main__":
    best_solution, best_cost = solve_aco()
    print_solution(best_solution, best_cost)
