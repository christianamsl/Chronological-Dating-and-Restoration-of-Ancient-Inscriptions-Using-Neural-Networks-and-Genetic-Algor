import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Διαβάζουμε τα δεδομένα από το αρχείο CSV, ορίζοντας τη μηχανή ως "python" 
data = pd.read_csv("iphi2802 (3).csv", sep="	", engine="python", header=0, encoding="utf-8")
data = data[data['region_main_id'] == 1683]  # Φιλτράρισμα για την περιοχή της Συρίας


# Προεπεξεργασία κειμένου
data['text'] = data['text'].apply(lambda x: x.lower()) #(μικρα=κεφαλαία)
data['tokens'] = data['text'].apply(word_tokenize)

# Δημιουργία tf-idf διανυσμάτων
vectorizer = TfidfVectorizer(tokenizer=word_tokenize , max_features=1678)
tfidf_matrix = vectorizer.fit_transform(data['text'])

# Λεξικό 
vocabulary = vectorizer.vocabulary_
vocab_size = len(vocabulary)

# Υλοποίηση Γενετικού Αλγορίθμου
population_size = 100
generations = 100
mutation_rate = 0.01

# Δημιουργία αρχικού πληθυσμού
def create_individual():
    return [random.randint(1, vocab_size), random.randint(1, vocab_size)] #(δημιουργία ενός ατόμου που περιέχει δύο τυχαίες λέξεις απο το λεξικό)

def create_population(size):
    return [create_individual() for _ in range(size)] #(δημιουργία πληθυσμού απο τα άτομα )

population = create_population(population_size)

# Υπολογισμός ομοιότητας με χρήση συνάρτησης cosine_similarity
def compute_similarity(individual, target_vector):
    missing_words = [list(vocabulary.keys())[individual[0]-1], list(vocabulary.keys())[individual[1]-1]]
    completed_text = " ".join([missing_words[0], "αλεξανδρε", "ουδις", missing_words[1]])
    completed_vector = vectorizer.transform([completed_text])
    return cosine_similarity(completed_vector, target_vector)[0][0]

# Επιλογή Top-K επιγραφών για αναφορά
def find_top_k_similar(data, target_vector, k=5):
    similarities = cosine_similarity(tfidf_matrix, target_vector)
    top_k_indices = np.argsort(similarities[:, 0])[-k:]
    return data.iloc[top_k_indices].reset_index(drop=True)

# Συνάρτηση καταλληλότητας
def fitness(individual, top_k_data):
    missing_words = [list(vocabulary.keys())[individual[0]-1], list(vocabulary.keys())[individual[1]-1]]
    completed_text = " ".join([missing_words[0], "αλεξανδρε", "ουδις", missing_words[1]])
    completed_vector = vectorizer.transform([completed_text])
    
    # Υπολογισμός μέσης ομοιότητας με τις top-K επιγραφές
    similarities = cosine_similarity(completed_vector, tfidf_matrix[top_k_data.index])
    return np.mean(similarities)


# Επιλογή (Selection)
def tournament_selection(population, fitnesses, tournament_size=3):
    new_population = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        new_population.append(winner)
    return new_population

# Διασταύρωση (Crossover)
def uniform_crossover(parent1, parent2):
    return [parent1[i] if random.random() > 0.5 else parent2[i] for i in range(len(parent1))]

#Μετάλλαξη (Mutate)
def mutate(individual):
    if random.random() < mutation_rate:
        individual[random.randint(0, 1)] = random.randint(1, vocab_size)
    return individual

def evolve(population, target_vector, top_k_data, elite_size=2):
    fitnesses = [fitness(ind, top_k_data) for ind in population]
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
    next_generation = sorted_population[:elite_size]
    selected_population = tournament_selection(population, fitnesses)
    for i in range(0, len(selected_population) - elite_size, 2):
        if i+1 < len(selected_population):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1 = mutate(uniform_crossover(parent1, parent2))
            child2 = mutate(uniform_crossover(parent2, parent1))
            next_generation.extend([child1, child2])
    return next_generation, max(fitnesses)

# Στόχος (Target) επιγραφή
target_text = "αλεξανδρε ουδις"
target_vector = vectorizer.transform([target_text])

# Top-5 παρόμοιες επιγραφές για αναφορά
top_k_data = find_top_k_similar(data, target_vector, k=5)
print("Top-5 παρόμοιες επιγραφές:")
for idx, text in enumerate(top_k_data['text']):
    print(f"{idx+1}. {text}")

'''
# Εξέλιξη του πληθυσμού για προκαθορισμένο αριθμό γενεών
best_fitness = 0
for generation in range(generations):
    population, best_fitness_in_generation = evolve(population, target_vector, top_k_data)
    if best_fitness_in_generation > best_fitness:
        best_fitness = best_fitness_in_generation
    if generation % 10 == 0:
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

# Καλύτερο άτομο του πληθυσμού
best_individual = max(population, key=lambda ind: fitness(ind, top_k_data))
missing_words = [list(vocabulary.keys())[best_individual[0]-1], list(vocabulary.keys())[best_individual[1]-1]]
completed_text = " ".join([missing_words[0], "αλεξανδρε", "ουδις", missing_words[1]])
print(f"Αποκαταστάθηκε η επιγραφή: {completed_text}")
'''
# Προκαθορισμένες παράμετροι
population_sizes = [20, 200]
crossover_probabilities = [0.6, 0.9, 0.1]
mutation_probabilities = [0.00, 0.01, 0.10]
num_runs = 10  # Αριθμός επαναλήψεων για κάθε συνδυασμό παραμέτρων
max_generations = 1000  # Μέγιστος αριθμός γενεών
stagnation_limit = 50  # Όριο στασιμότητας βελτίωσης

# Κύρια συνάρτηση εκτέλεσης

def run_ga(population_size, crossover_prob, mutation_prob, num_runs, max_generations, stagnation_limit):
    global mutation_rate, crossover_rate, vocab_size , generations
    mutation_rate = mutation_prob
    crossover_rate = crossover_prob
    vocab_size = len(vocabulary)

    best_fitness_overall = []
    num_generations_overall = []
    best_individual_overall = None
    fitness_over_generations = np.zeros((num_runs, max_generations)) # προσθέσαμε για τα διαγράμματα

    for run in range(num_runs):
        population = create_population(population_size)
        best_fitness = 0
        best_fitness_stagnation = 0
        best_individual = None

        for generation in range(max_generations):
            population, current_best_fitness = evolve(population, target_vector, top_k_data)
            fitness_over_generations[run, generation] = current_best_fitness # για τα διαγράμματα 

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_fitness_stagnation = 0
                best_individual = population[np.argmax(np.sum(population, axis=1))]  # Βρείτε το καλύτερο άτομο
            else:
                best_fitness_stagnation += 1

            if best_fitness_stagnation > stagnation_limit or current_best_fitness < best_fitness * 1.01:
                break

        best_fitness_overall.append(best_fitness)
        num_generations_overall.append(generation + 1)
        if best_individual_overall is None or best_fitness > np.sum(best_individual_overall):
            best_individual_overall = best_individual

    mean_fitness_over_generations = np.mean(fitness_over_generations, axis=0)
    return np.mean(best_fitness_overall), np.mean(num_generations_overall), best_individual_overall , mean_fitness_over_generations



# Εκτέλεση για όλους τους συνδυασμούς παραμέτρων
results = []

best_individual_global = None
best_fitness_global = -np.inf

for population_size in population_sizes:
    for crossover_prob in crossover_probabilities:
        for mutation_prob in mutation_probabilities:
            mean_fitness, mean_generations, best_individual ,  mean_fitness_over_generations = run_ga(population_size, crossover_prob, mutation_prob, num_runs, max_generations, stagnation_limit)
            results.append((population_size, crossover_prob, mutation_prob, mean_fitness, mean_generations, best_individual , mean_fitness_over_generations))

            if best_individual is not None and mean_fitness > best_fitness_global:
                best_fitness_global = mean_fitness
                best_individual_global = best_individual
            # Δημιουργία της λίστας generations εντός του βρόγχου για την κατάλληλη διάσταση
           # generations = range(1, len(mean_fitness_over_generations) + 1)

           
# Εκτύπωση αποτελεσμάτων
for idx, (population_size, crossover_prob, mutation_prob, mean_fitness, mean_generations, mean_fitness_over_generations, best_individual) in enumerate(results):
    print(f"{idx+1}\t{population_size}\t{crossover_prob}\t{mutation_prob}\t{mean_fitness:.4f}\t{mean_generations:.2f}")

# Εκτύπωση του καλύτερου ατόμου συνολικά
if best_individual_global is not None:
    missing_words = [list(vocabulary.keys())[list(vocabulary.values()).index(best_individual_global[0])],
                     list(vocabulary.keys())[list(vocabulary.values()).index(best_individual_global[1])]]
    completed_text = " ".join([missing_words[0], "αλεξανδρε", "ουδις", missing_words[1]])
    print(f"Αποκαταστάθηκε η επιγραφή: {completed_text}")

