import random
import numpy as np
import argparse

def extract_best(population):
  best_value = 0
  best_chromosome = []
  for chromosome in population:
    total_value = np.dot(chromosome, values)
    if total_value > best_value:
      best_chromosome = chromosome
      best_value = total_value
  return best_value, best_chromosome

def mutation(children_chromosomes):
  for chromosomes in children_chromosomes:
    mutation_probability = random.uniform(0, 1)
    if mutation_probability < PROB_UNIVERSAL:
      mutation_point = random.randint(0, num_items-1)
      if chromosomes[mutation_point] == 0:
        chromosomes[mutation_point] = 1
      else:
        chromosomes[mutation_point] = 0

  return children_chromosomes

def crossover(parents_chromosomes):
  children_chromosomes = []
  num_couples = int(len(parents_chromosomes) / 2)
  for i in range(num_couples):
    parent1 = 2*i
    parent2 = 2*i + 1
    crossover_point = random.randint(1, num_items-1)
    child1 = np.append(parents_chromosomes[parent1][:crossover_point],
                          parents_chromosomes[parent2][crossover_point:])
    children_chromosomes.append(child1)
    child2 = np.append(parents_chromosomes[parent2][:crossover_point],
                          parents_chromosomes[parent1][crossover_point:])
    children_chromosomes.append(child2)

  return children_chromosomes

def selected_parent(fitness_items_probability):
  random_probability = random.uniform(0, 1)
  i = 0
  while random_probability > fitness_items_probability[i]:
    i += 1
  return i


def select_parents(population, number_parents):
  fitness_items = list(map(fitness, population))
  fitness_items_sorted = sorted(fitness_items, reverse=True)
  fitness_sum = sum(fitness_items_sorted)
  fitness_items_probability = list(map(lambda x: x / fitness_sum, fitness_items_sorted))
  num_probability = len(fitness_items_probability)

  for i in range(1, num_probability):
    fitness_items_probability[i] += fitness_items_probability[i-1]
  fitness_items_probability[num_probability-1] = 1

  for_mapping = np.argsort(fitness_items)[::-1]

  parent_indexes = []
  while len(parent_indexes) < number_parents:
    parent_index = selected_parent(fitness_items_probability)
    if(parent_index not in parent_indexes):
      parent_indexes.append(parent_index)

  parents_chromosomes = []
  for parent in parent_indexes:
    parents_chromosomes.append(population[for_mapping[parent]])

  return parents_chromosomes


def next_population(population):
  parents_chromosomes = select_parents(population, len(population)/2)
  children_chromosomes = crossover(parents_chromosomes)
  children_mutation_chromosomes = mutation(children_chromosomes)

  valid_children_chromosomes = []
  for chromosome in children_mutation_chromosomes:
    if is_valid_chromosome(chromosome):
      valid_children_chromosomes.append(chromosome)
    else:
      valid_children_chromosomes += initial_population(1)

  new_population = parents_chromosomes + valid_children_chromosomes
  return new_population

def fitness(chromosome):
  return np.dot(chromosome, values)

def is_valid_chromosome(chromosome):
  total_weight = np.dot(chromosome, weights)
  return total_weight < capacity

def initial_population(n):
  population = []
  while len(population) < n:
    chromosome = np.random.randint(2, size=num_items)
    if(is_valid_chromosome(chromosome)):
      population.append(chromosome)
  return population

parser = argparse.ArgumentParser(description='Get file directory')
parser.add_argument('file_dir', help='file directory', type=str)
args = parser.parse_args()

with open(args.file_dir, 'r') as f:
  lines = f.readlines()

num_items, capacity = lines[0].replace('\n', '').split(' ')
num_items = int(num_items)
capacity  = int(capacity)

lines.pop(0)

values = []
weights = []

for line in lines:
  value, weight = line.replace('\n', '').split(' ')
  values.append(float(value))
  weights.append(float(weight))

iterations = 1000
BEST_VALUE_GLOBAL = 0
BEST_CHROMOSOME_GLOBAL = []
BEST_ITERATION = 0
PROB_UNIVERSAL = 0

population = initial_population(20)
for i in range(iterations):
  best_value, best_chromosome = extract_best(population)
  if best_value > BEST_VALUE_GLOBAL:
    BEST_CHROMOSOME_GLOBAL = best_chromosome
    BEST_VALUE_GLOBAL = best_value
    BEST_ITERATION = i
  print('\n{}º iteration\nBest Value: {}\tChromosome: {}'.format(i+1, BEST_VALUE_GLOBAL, BEST_CHROMOSOME_GLOBAL))
  population = next_population(population)
print("\bBest iteration: {}\t Best value: {}".format(BEST_ITERATION, BEST_VALUE_GLOBAL))
