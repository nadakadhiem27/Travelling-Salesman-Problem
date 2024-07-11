#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import random


# In[22]:


import random

# Define the number of cities
num_cities = 6

# Function to generate a random population
def generate_population(population_size):
    population = []
    for _ in range(population_size):
        # Generate a random chromosome (route)
        chromosome = random.sample(range(num_cities), num_cities)
        population.append(chromosome)
    return population

# Example usage
population_size = 6
population = generate_population(population_size)
print("Random Population:")
for i, chromosome in enumerate(population):
    print(f"Chromosome {i+1}: {chromosome}")


# make fitness fuction take chromosom and calculate summation of distance between city in each chromosom
# *Fitness Calculation
# Evaluate the fitness of each path. The fitness is determined by the total distance of the route. Shorter routes have higher fitness.

# In[27]:


# Function to calculate the Euclidean distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate the total distance of a route (chromosome)
def route_distance(chromosome, cities):
    total_distance = 0
    num_cities = len(chromosome)
    for i in range(num_cities):
        city1 = cities[chromosome[i]]
        city2 = cities[chromosome[(i + 1) % num_cities]]  # Connect last city with the first city
        total_distance += distance(city1, city2)
    return total_distance
cities = [(0,0),(4,1),(1,3),(3,2),(2,5),(5,6)]

# Fitness calculation for each chromosome in the population
fitness_scores = [route_distance(chromosome, cities) for chromosome in population]
print("Fitness Scores:")
for i, fitness in enumerate(fitness_scores):
    print(f"Chromosome {i+1} Fitness: {fitness}")

result = sum(fitness_scores)
print("Summation of Fitness Scores:", result)


# In[28]:


best_index = np.argmin(fitness_scores)
best_chromosome = population[best_index]
best_fitness = fitness_scores[best_index]
print("Best Chromosome:", best_chromosome)
print("Best Fitness:", best_fitness)


# Tournament Selection 
# Algorithm --
# 1.Select k individuals from the population and perform a tournament amongst them
# 2.Select the best individual from the k individuals
# 3. Repeat process 1 and 2 until you have the desired amount of population

# In[43]:


def tournament_selection(population, fitness_scores, tournament_size):
    selected_parents = []
    population_size = len(population)
    for _ in range(population_size):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(population_size), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        # Choose the winner of the tournament (individual with the highest fitness)
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected_parents.append(population[winner_index])
    return selected_parents
tournament_size = 2

# Tournament selection
selected_parents = tournament_selection(population, fitness_scores, tournament_size)
print("Selected Parents:")
for i, parent in enumerate(selected_parents):
    print(f"Parent {i+1}: {parent}")


# In[54]:


# Function for tournament selection
def tournament_selection(population, fitness_scores, k):
    selected_parents = []
    population_size = len(population)
    while len(selected_parents) < population_size:
        # Select k individuals randomly from the population for the tournament
        tournament_indices = random.sample(range(population_size), k)
        # Choose the best individual from the tournament
        best_fitness = float('inf')  # Initialize with infinity to ensure any fitness will be smaller
        best_index = None
        for i in tournament_indices:
            if fitness_scores[i] < best_fitness:  # Compare fitness scores directly
                best_fitness = fitness_scores[i]
                best_index = i
        selected_parents.append(population[best_index])
    return selected_parents


selected_parents = tournament_selection(population, fitness_scores, 2)
print("Selected Parents:")
for i, parent in enumerate(selected_parents):
    print(f"Parent {i+1}: {parent}")


# Crossover
# Apply crossover (recombination) to the selected parents to create offspring, use one point crossover , in TSP permution is  take number untel point from parent 1 , then the second parent is scanned and if number is not yet in the offspring it is added 

# In[65]:


import random

# Function for one-point crossover
def one_point_crossover(parent1, parent2):
    crossover_point = 3

    # Create an empty offspring
    offspring = []

    # Take numbers until the crossover point from parent 1
    for i in range(crossover_point):
        offspring.append(parent1[i])

    # Scan parent 2 and add numbers to the offspring if not already present
    for num in parent2:
        if num not in offspring:
            offspring.append(num)

    return offspring



parent1 = random.choice(population)
parent2 = random.choice(population)

# Perform one-point crossover
offspring = one_point_crossover(parent1, parent2)
print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("Offspring after one-point crossover:", offspring)


# Mutation
# Apply mutation to the offspring. This involves making small random changes to some of the genes (cities) in the population.

# In[96]:


import random

def mutation(chrom, mutation_prop):
    mutated_chrom = chrom[:]  # Create a copy of the chromosome to avoid modifying the original
    for i in range(len(mutated_chrom)):
        if random.random() < mutation_prop:
            # Select a random index to swap with (excluding the current index)
            swap_index = random.choice([index for index in range(len(mutated_chrom)) if index != i])
            # Swap cities at the current index and the selected index
            mutated_chrom[i], mutated_chrom[swap_index] = mutated_chrom[swap_index], mutated_chrom[i]
    return mutated_chrom

# Example usage
offspring = [2, 3, 0, 1, 5, 4]
mutation_prop = 0.1  # Mutation probability

print("Offspring before mutation:", offspring)
result = mutation(offspring, mutation_prop)
print("Mutated offspring:", result)


# In[103]:


import random

def mutation(chrom, mutation_prop):
    mutated_chrom = chrom[:]  # Create a copy of the chromosome to avoid modifying the original
    for i in range(len(mutated_chrom)):
        if random.random() < mutation_prop:
            # Select a random index to swap with
            swap_index = random.randint(0, len(mutated_chrom) - 1)
            # Swap cities at the current index and the selected index
            mutated_chrom[i], mutated_chrom[swap_index] = mutated_chrom[swap_index], mutated_chrom[i]
    return mutated_chrom


mutation_prop = 0.1  # Mutation probability

print("Offspring before mutation:", offspring)
result = mutation(offspring, mutation_prop)
print("Mutated offspring:", result)


# In[ ]:




