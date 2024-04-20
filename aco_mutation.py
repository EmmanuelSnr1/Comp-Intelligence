import numpy as np
import random
import logging

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Problem definition
stock_lengths = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
stock_costs = [86, 85, 83, 79, 68, 66, 64, 63]
piece_lengths = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
quantities = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

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
    #logging.debug(f"Post-mutation solution: {solution}")

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

        # Update pheromones
        update_pheromones(pheromones, solutions, best_cost)

        # Mutation step
        for solution, cost in solutions:
            mutate(solution, current_mutation_rate)

        # Adjust mutation rate based on performance
        current_mutation_rate = adjust_mutation_rate(best_cost, cost, current_mutation_rate)
        #logging.debug(f"Current mutation rate: {current_mutation_rate}")

    return best_solution, best_cost


def construct_solution(pheromones, remaining_quantities):
    solution = []
    remaining_quantities = remaining_quantities[:]
    for stock_index in range(len(stock_lengths)):
        activities = []  # List of activities for this stock type
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
                    # Update pheromones for each piece used in this activity
                    pheromones[stock_index][piece_index] += 1 / (cost + 1)
    pheromones *= (1 - decay)
    
    
def calculate_fitness(solution):
    cost = 0
    for stock_index, activities in solution:
        for activity in activities:
            if activity:  # Check if this activity is non-empty
                cost += stock_costs[stock_index]  # Each activity uses a new stock piece
    return cost


def print_solution(solution, cost):
    print(f"Best Cost: {cost}")
    print("Solution:")
    for stock_index, activities in solution:
        print(f"Stock Length {stock_lengths[stock_index]}:")
        for activity in activities:
            pieces = [piece_lengths[piece_index] for piece_index in activity]
            print(f"  Pieces cut: {pieces}")
            print(f"  Total length used: {sum(pieces)} / {stock_lengths[stock_index]}")
        print()  # Adds a new line for better separation

# Main execution block
if __name__ == "__main__":
    best_solution, best_cost = solve_aco()
    print_solution(best_solution, best_cost)
