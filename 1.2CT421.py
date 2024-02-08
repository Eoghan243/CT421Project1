import random
import matplotlib.pyplot as plt

def generate_individual(length):
    return [random.choice(['0', '1']) for _ in range(length)]

def calculate_fitness(individual, target_string):
    return sum([1 for i, gene in enumerate(individual) if gene == target_string[i]])

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    mutated_individual = [bit if random.random() > mutation_rate else flip_bit(bit) for bit in individual]
    return mutated_individual

def flip_bit(bit):
    return '0' if bit == '1' else '1'

def evolve_to_target(target_string, population_size, mutation_rate, generations):
    population = [generate_individual(len(target_string)) for _ in range(population_size)]

    fitness_history = []

    for generation in range(generations):
        fitness_scores = [calculate_fitness(individual, target_string) for individual in population]
        average_fitness = sum(fitness_scores) / len(population)
        fitness_history.append(average_fitness)
        print(f"Generation {generation + 1}: Average Fitness = {average_fitness}")

        if target_string in [''.join(individual) for individual in population]:
            print("Target string found!")
            break

        parents = random.choices(population, weights=fitness_scores, k=population_size)

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation

    # Plot the fitness history
    plt.plot(range(1, len(fitness_history) + 1), fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Evolutionary Algorithm Fitness Progress')
    plt.show()

if __name__ == "__main__":
    target_string = "110010101010101010100101010101"
    population_size = 100
    mutation_rate = 0.02
    generations = 80

    evolve_to_target(target_string, population_size, mutation_rate, generations)
