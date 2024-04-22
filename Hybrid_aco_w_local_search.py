
import numpy as np
import random
import logging
import time



# # Problem Definition
# stock_lengths = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
# stock_costs = [86, 85, 83, 79, 68, 66, 64, 63]
# piece_lengths = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# quantities = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

stock_lengths = [120, 115, 110, 105, 100]
stock_costs = [12, 11.5, 11, 10.5, 10]
piece_lengths = [21, 22, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66, 67]
quantities = [13, 15, 7, 5, 9, 9, 3, 15, 18, 17, 4, 17, 20, 9, 4, 19, 4, 12, 15, 3, 20, 14, 15, 6, 4, 7, 5, 19, 19, 6, 3, 7, 20, 5, 10, 17]


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Parameters
num_ants = 10
num_iterations = 10
alpha = 1.0  # Pheromone influence
beta = 1.0   # Heuristic influence
decay = 0.1  # Pheromone decay rate
initial_pheromone = 0.1

# Initialize pheromones
pheromones = np.full((len(stock_lengths), len(piece_lengths)), initial_pheromone)

# Update heuristic to consider current stock usage
def heuristic_value(stock_index, piece_index):
    noise = random.uniform(0.9, 1.1)
    piece_utilization = piece_lengths[piece_index] / stock_lengths[stock_index] * noise
    return piece_utilization / stock_costs[stock_index]

def apply_local_search(solution, remaining_quantities):
    improved = True
    while improved:
        improved = False
        for stock_index, activities in solution:
            for pieces in activities:  # Now correctly handling 'pieces' as each individual activity
                if not pieces:
                    continue
                gaps = stock_lengths[stock_index] - sum(piece_lengths[p] for p in pieces)
                for j in range(len(piece_lengths)):
                    if piece_lengths[j] <= gaps and remaining_quantities[j] > 0:
                        pieces.append(j)
                        remaining_quantities[j] -= 1
                        gaps -= piece_lengths[j]
                        improved = True

                # Try rearranging pieces within and between stocks to minimize the number of stocks used
                for other_index, other_activities in enumerate(solution):
                    if stock_index == other_index:
                        continue
                    for other_pieces in other_activities:
                        other_gaps = stock_lengths[other_index] - sum(piece_lengths[p] for p in other_pieces)
                        for piece in list(pieces):
                            if piece_lengths[piece] <= other_gaps:
                                # Move piece to another stock
                                other_pieces.append(piece)
                                pieces.remove(piece)
                                other_gaps -= piece_lengths[piece]
                                gaps += piece_lengths[piece]
                                improved = True
                                break  # Reevaluate after each move
    return solution



def update_pheromones(pheromones, solutions, best_cost):
    for solution, cost in solutions:
        for stock_index, activities in solution:
            for activity in activities:
                for piece_index in activity:
                    # Update pheromones for each piece used in this activity
                    pheromones[stock_index][piece_index] += 1 / (cost + 1)
    pheromones *= (1 - decay)


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


def calculate_fitness(solution):
    cost = 0
    for stock_index, activities in solution:
        for activity in activities:
            if activity:  # Check if this activity is non-empty
                cost += stock_costs[stock_index]  # Each activity uses a new stock piece
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
    best_solution, best_cost = solve()
    end_time = time.time()
    computation_time = end_time - start_time
    total_waste = calculate_waste(solution)
    print("Total Waste:", total_waste)
    print("Computation time :", computation_time)
    print(f"Best Cost: {cost}")
    print("Solution:")
    for stock_index, activities in solution:
        for activity in activities:
            pieces = [piece_lengths[piece_index] for piece_index in activity]
            print(f"Stock Type {stock_index} (Length {stock_lengths[stock_index]}): Pieces cut: {pieces}")


def solve():
    best_solution = None
    best_cost = float('inf')
    for _ in range(num_iterations):
        solutions = []
        for _ in range(num_ants):
            remaining_quantities = quantities[:]
            logging.debug(f"Starting quantities for ant: {remaining_quantities}")
            solution = construct_solution(pheromones, remaining_quantities)
            logging.debug(f"Solution before local search: {solution}")
            solution = apply_local_search(solution, remaining_quantities)
            logging.debug(f"Solution after local search: {solution}")
            cost = calculate_fitness(solution)
            logging.debug(f"Cost of solution: {cost}")
            solutions.append((solution, cost))
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
        update_pheromones(pheromones, solutions, best_cost)
    return best_solution, best_cost

if __name__ == "__main__":
    best_solution, best_cost = solve()
    print_solution(best_solution, best_cost)
