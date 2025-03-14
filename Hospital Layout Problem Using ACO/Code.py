import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class ACO_Hospital_layout_Solver:
    def __init__(self, patient_flow_file, distance_matrix_file, alpha, beta, rho, total_ants, Q, iterations):
        # Read data from files
        self.patient_flow_matrix, self.distance_matrix = self.read_data(patient_flow_file, distance_matrix_file)
        
        # Get dimensions
        n_facilities = len(self.patient_flow_matrix)
        n_locations = len(self.distance_matrix)

        # Initialize desirability matrix
        self.desirability_matrix = np.zeros((n_facilities, n_locations))
        self.Calculate_desirability(n_facilities, n_locations)

        # Initialize intensity matrix (pheromone levels)
        self.intensity_matrix = np.ones((n_facilities, n_locations)) * 1e-6

        # Run ACO algorithm
        self.Perform_ACO(Q, alpha, beta, rho, total_ants, iterations, n_facilities, n_locations)
        
    def read_matrix(self, file_path):
        """
        Helper function to read a matrix from a text file.
        """

        return np.loadtxt(file_path, dtype = int)

    def read_data(self, patient_flow_file, distance_matrix_file):
        """
        Reads patient flow and distance matrix data from text files and returns them as 2D arrays.

        :param patient_flow_file: Path to the file containing patient flow data.
        :param distance_matrix_file: Path to the file containing distance matrix data.
        :return: Two 2D arrays (lists of lists): patient_flow_matrix, distance_matrix.
        """
    
        # Read patient flow matrix
        patient_flow_matrix = self.read_matrix(patient_flow_file)

        # Read distance matrix
        distance_matrix = self.read_matrix(distance_matrix_file)
        return patient_flow_matrix, distance_matrix
    
    def Random_population_generator(self, total_facilities = 19 , total_locations = 19):
        # Generate a NumPy array with distinct values from 0 to 18
        return np.random.choice(total_facilities, size=total_locations, replace=False)  # Adjust size as needed
    
    def Show_Actual_Solution(self, solution):
        return (np.array(solution) + 1) #since our actual solutions are sequenced integer from 1 to 19 not index from 0 to 18.

    def Calculate_population_score(self, representation):
        score = 0
        n = len(representation)
        
        for i in range(n):
            current_facility = representation[i]
            for j in range(n):
                if i==j: continue

                next_facility = representation[j]  # Wrap around to the first facility
                
                # Calculate the distance between the current and next facility
                distance = self.distance_matrix[i, j]
                
                # Calculate the patient flow between the current and next facility
                flow = self.patient_flow_matrix[current_facility, next_facility]
                
                # Add to the total score
                score += flow * distance
        
        return score
    
    def Calculate_desirability(self, n_facilities, n_locations, best_solution=None):
        """
        Calculate the desirability matrix for assigning facilities to locations.
        If a best_solution is provided, update the desirability matrix dynamically.
        """
        self.desirability_matrix = np.zeros((n_facilities, n_locations))

        for i in range(n_facilities):  # For each facility
            for j in range(n_locations):  # For each location
                total_cost = 0
                for k in range(n_facilities):  # For all other facilities
                    if k != i:
                        # Find the location of facility k in the best solution
                        if best_solution is not None:
                            loc_k = best_solution[k]  # Use the best solution to find the location of facility k
                        else:
                            loc_k = k  # Default to initial assumption (e.g., facilities assigned in order)

                        # Calculate the cost contribution of facility i at location j
                        flow = self.patient_flow_matrix[i][k]
                        distance = self.distance_matrix[j][loc_k]
                        total_cost += flow * distance

                # Avoid division by zero
                if total_cost == 0:
                    self.desirability_matrix[i][j] = np.inf
                else:
                    self.desirability_matrix[i][j] = 1 / total_cost

        # If a best solution is provided, reinforce the desirability of assignments in the best solution
        if best_solution is not None:
            for location, facility in enumerate(best_solution):
                self.desirability_matrix[facility][location] *= 1.5  # Increase desirability for assignments in the best solution


    def Update_trail_intensity(self, Q, representations, rho):
        """
        Updates the pheromone matrix based on the solutions found by the ants.

        :param Q: A constant for pheromone update.
        :param representations: A list of tuples, where each tuple contains:
                                - solution: A permutation of facilities assigned to locations.
                                - score: The quality (cost) of the solution.
        :param rho: The evaporation rate.
        """
        # Apply evaporation to all pheromone values
        self.intensity_matrix *= (1 - rho)
        
        # Update pheromone levels based on the solutions
        for solution, score in representations:
            if score == 0:  # Avoid division by zero
                continue
            
            # Iterate over the solution to get facility-location pairs
            for location, facility in enumerate(solution):
                self.intensity_matrix[facility, location] += Q / score

        
    def Calculate_probability_at_time_t(self, alpha, beta):
        n_facilities = len(self.intensity_matrix)
        n_locations = len(self.intensity_matrix[0])
        probability_matrix = np.zeros((n_facilities, n_locations), dtype=float)

        # It avoid division by zero henc runtime warning
        epsilon = 1e-10
        
        for i in range(n_facilities):
            # Calculating numerator: tau^alpha * eta^beta
            numerator = (self.intensity_matrix[i] ** alpha) * (self.desirability_matrix[i] ** beta)
            
            # Checking for NaN/Inf in numerator and handle
            if np.any(np.isnan(numerator)) or np.any(np.isinf(numerator)):
                # For Invalid values in numerator for facility {i} we are replacing with epsilon.
                numerator = np.ones_like(numerator) * epsilon  # Fallback to uniform probability
                
            denominator = np.sum(numerator)
            
            # If denominator is zero/NaN/Inf, we will use uniform probabilities
            if denominator <= epsilon or np.isnan(denominator) or np.isinf(denominator):
                # For Invalid denominator for facility {i}. We are using uniform probabilities.")
                probability_matrix[i] = np.ones(n_locations) / n_locations  # Uniform distribution
            else:
                probability_matrix[i] = numerator / (denominator + epsilon)  # Adding epsilon for stability
                
        return probability_matrix


    def Calculate_next_facility(self, probability_matrix, current_location): #rank based selection
        # Extract probabilities for the current location
        facility_probabilities = probability_matrix[:, current_location]
    
        # Rank facilities in descending order of probability
        ranks = np.argsort(facility_probabilities)[::-1]  # Descending order
    
        # Assign rank-based probabilities (linear ranking)
        rank_probabilities = np.arange(len(ranks), 0, -1)  # Linear ranking: [n, n-1, ..., 1]
        rank_probabilities = rank_probabilities / np.sum(rank_probabilities)  # Normalize
    
        # Perform roulette wheel selection based on rank probabilities
        r = np.random.rand()
        cumulative = np.cumsum(rank_probabilities)
        selected_rank = np.argmax(cumulative >= r)  # Find the selected rank
        selected_facility = ranks[selected_rank]  # Map rank back to facility index

        return selected_facility
    
    def Plot_scores(self, best_fitness, avg_fitness, alpha, beta, rho, total_ants, total_iterations):
        # Create a figure
        plt.figure(figsize=(10, 6))

        # Plot Best Fitness
        plt.plot(range(len(best_fitness)), best_fitness, label='Best Fitness', color='green', linewidth=2)

        # Plot Average Fitness
        plt.plot(range(len(avg_fitness)), avg_fitness, label='Average Fitness', color='blue', linestyle='--', linewidth=2)

        # Add labels, title, and grid
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness (Cost)', fontsize=12)
        plt.title(f'ACO Optimization Progress\nAlpha = {alpha}, Beta = {beta}, Rho = {rho}, Ants = {total_ants}, Iterations = {total_iterations}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        plt.legend(fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()       
    
    def Perform_ACO(self, Q, alpha, beta, rho, total_ants, total_iterations, n_facilities, n_locations):
        # Initialize global best solution
        global_best_solution = None
        global_best_score = float('inf')
        iterations_best_solution = []
        iterations_avg_fitness = []

        # Initialize using Random Population Generations
        representations = []
        temp = []
        for i in range(total_ants):
            sol = self.Random_population_generator(19, 19)
            if len(representations) >= 19:
                representations.append((sol.tolist(), self.Calculate_population_score(sol)))
            else:
                while sol[0] in temp:
                    sol = self.Random_population_generator(19, 19)
                score = self.Calculate_population_score(np.array(sol))
                representations.append((sol, score))
                temp.append(sol[0])

        # Calculate initial desirability matrix
        self.Calculate_desirability(n_facilities, n_locations)

        # Update pheromone trail intensity matrix
        self.Update_trail_intensity(Q, representations, rho)
        global_best_solution, global_best_score = min(representations, key=lambda x: x[1])
        iterations_best_solution.append(global_best_score)

        # Calculate average fitness for this iteration
        current_avg_score = np.mean([score for (sol, score) in representations])
        iterations_avg_fitness.append(current_avg_score)

        # Run for the remaining iterations
        for iteration in range(total_iterations - 1):
            representations = []
            for ant in range(total_ants):
                if ant <= 18:  # Ensure at least one solution starts from every location
                    j = ant
                else:
                    j = np.random.randint(0, 19)

                sol = [j]  # First element of solution
                while len(sol) < n_locations:
                    next_facility = self.Calculate_next_facility(self.Calculate_probability_at_time_t(alpha, beta), len(sol))
                    if next_facility not in sol:
                        sol.append(next_facility)
                    else:
                        remaining = [loc for loc in range(n_locations) if loc not in sol]
                        if not remaining:
                            break
                        sol.append(np.random.choice(remaining))

                if len(sol) != n_locations:
                    continue
                score = self.Calculate_population_score(np.array(sol))
                representations.append((sol, score))

            # Update pheromone trail intensity matrix
            self.Update_trail_intensity(Q, representations, rho)

            # Update desirability matrix based on the current best solution
            current_best_solution, current_best_score = min(representations, key=lambda x: x[1])
            self.Calculate_desirability(n_facilities, n_locations, current_best_solution)

            # Update global best solution
            if current_best_score < global_best_score:
                global_best_solution = current_best_solution
                global_best_score = current_best_score

            # Track best and average fitness
            iterations_best_solution.append(global_best_score)
            current_avg_score = np.mean([score for (sol, score) in representations])
            iterations_avg_fitness.append(current_avg_score)

        print(f"Global Best Solution: {self.Show_Actual_Solution(global_best_solution)}, Score: {int(global_best_score)}")
        self.Plot_scores(iterations_best_solution, iterations_avg_fitness, alpha, beta, rho, total_ants, total_iterations)

if __name__ == "__main__":
    # ACO_Hospital_layout_Solver(patient_flow_file, distance_matrix_file, alpha, beta, rho, total_ants, Q, iterations)
    obj = ACO_Hospital_layout_Solver(
        "data_facilities_patient_flow.txt", 
        "data_location_distances.txt", 
        alpha=3, 
        beta=4, 
        rho=0.3, 
        total_ants=105, 
        Q=1000, 
        iterations=2000
    )