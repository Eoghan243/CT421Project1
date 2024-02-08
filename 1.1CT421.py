import random
import matplotlib.pyplot as plt

population_size = 100
dna_size = 30
mutation_rate = 0.2
generations = 100  

def create_individual():
    return [random.randint(0, 1) for _ in range(dna_size)]

def calculate_fitness(individual):
    return sum(individual)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, dna_size - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual):
    for i in range(dna_size):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

def evolve_population(population):
    next_generation = []
    
    #
    preserve_count = int(population_size * 0.1)  # 10% of the population
    next_generation.extend(random.sample(population, preserve_count))
    
    while len(next_generation) < population_size:
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        next_generation.append(child1)
        next_generation.append(child2)
    
    return next_generation

def average_fitness(population):
    total_fitness = sum(calculate_fitness(individual) for individual in population)
    return total_fitness / len(population)

def main():
    population = [create_individual() for _ in range(population_size)]
    avg_fitnesses = []
    for generation in range(1, generations + 1):
        avg_fit = average_fitness(population)
        avg_fitnesses.append(avg_fit)
        print(f"Generation {generation}: Average Fitness = {avg_fit}")
        population = evolve_population(population)
        if dna_size in [calculate_fitness(individual) for individual in population]:
            print("Solution found!")
            break
    
    if dna_size not in [calculate_fitness(individual) for individual in population]:
        print("No solution found.")
    
    # Plotting
    plt.plot(range(1, generation + 1), avg_fitnesses)
    plt.title('Average Fitness vs Generations')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.show()

if __name__ == "__main__":
    main()
