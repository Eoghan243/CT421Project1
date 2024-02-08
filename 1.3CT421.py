import numpy as np 
import matplotlib.pyplot as plt 

# Calculate fitness
def fitness_function(individual):
    if sum(individual) == 0:
        return 2 * len(individual)
    else: 
        return sum(individual)
    
# One-point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Mutation
def mutate(individual, mutation_rate, chromosome_length):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]


    if len(mutated_individual) < chromosome_length:
            mutated_individual = np.concatenate((mutated_individual, np.zeros(chromosome_length - len(mutated_individual), dtype=int)))
    elif len(mutated_individual) > chromosome_length:
            mutated_individual = mutated_individual[:chromosome_length]
    return mutated_individual
        
# Genetic algorithm
def genetic_algorithm(population_size, chromosome_length, generations, crossover_rate, mutation_rate):
    population = np.random.randint(0, 2, size=(population_size, chromosome_length))
    avg_fitness_over_time = []

    for generation in range(generations):
        # Calculate fitness for each individual
        fitness_values = np.array([fitness_function(individual) for individual in population])

        selected_indices = np.random.choice(range(population_size), size=population_size, replace=True)
        selected_population = population[selected_indices]

        # Do crossover
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                max_len = max(len(child1), len(child2))
                child1 = np.concatenate((child1, np.zeros(max_len - len(child1), dtype=int)))
                child2 = np.concatenate((child2, np.zeros(max_len - len(child2), dtype=int)))
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_population.extend([child1, child2])

        # Do mutation
        for i in range(len(new_population)):
            new_population[i] = mutate(new_population[i], mutation_rate, chromosome_length)

        population = np.array(new_population)

        # Calculate and store average fitness
        avg_fitness = np.mean(fitness_values)
        avg_fitness_over_time.append(avg_fitness)

        print(f"Generation {generation + 1}: Average Fitness = {avg_fitness}")

    return avg_fitness_over_time

population_size = 50  
chromosome_length = 10
generations = 100
crossover_rate = 0.8
mutation_rate = 0.01
item_count = 20
bin_capacity = 80

# Run the genetic algorithm
avg_fitness_over_time = genetic_algorithm(population_size, chromosome_length, generations, crossover_rate, mutation_rate )

# Print the lengths
print(len(range(1, generations + 1)))
print(len(avg_fitness_over_time))

# Plot results
plt.plot(range(1, generations + 1), avg_fitness_over_time, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()








