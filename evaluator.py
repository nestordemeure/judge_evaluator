import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path

# folder with the input jsons
input_folder = Path("./data")

# on how many questions are two models compared in a trial
nb_questions_per_trial = 20

# how many trials are conducted before picking our best, last, winner chatbot model?
# default to number of chatbot models
nb_trials = 10

# output files
output_folder = Path("./outputs")
alignement_plot_file = output_folder / "model_alignment.png"

#--------------------------------------------------------------------------------------------------
# IMPORT DATA

def import_jsons(input_folder:Path):
    # Initialize an empty dictionary to hold the data
    data_dict = {}

    # Iterate over all json files in the input folder
    for json_file in input_folder.glob('*_contest_data.json'):
        # Extract the judge name part of the filename
        name = json_file.stem.split('_contest_data')[0]
        
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # Add the data to the dictionary with the judge name as the key
        data_dict[name] = data
    
    return data_dict

# import the contests data to be analyzed
contests_data = import_jsons(input_folder)

#--------------------------------------------------------------------------------------------------
# COMPUTE CONTESTANTS LIST

def extract_contestants(contests_data):
    # Initialize a set to hold all contestant names
    contestants = set()

    # Collect contestant names from the contests_data
    for matches in contests_data.values():
        for match in matches:
            contestants.add(match['first'])
            contestants.add(match['second'])

    # Convert the set of contestants to a sorted list
    contestants = sorted(contestants)

    # Step 2: Create a mapping from contestant names to indices
    contestant_index = {name: i for i, name in enumerate(contestants)}
    return contestant_index

# extract contestants
contestants_index = extract_contestants(contests_data)
print(f"Contestants: {contestants_index}")

#--------------------------------------------------------------------------------------------------
# COMPUTE WIN PROBABILITY

def compute_win_probability(contestants_index, contests):
    # Initialize a matrix for win counts and match counts
    num_contestants = len(contestants_index)
    win_matrix = np.zeros((num_contestants, num_contestants))
    match_matrix = np.zeros((num_contestants, num_contestants))

    # Populate the win and match matrices
    for match in contests:
        first_idx = contestants_index[match['first']]
        second_idx = contestants_index[match['second']]
        
        # Increment the match count
        match_matrix[first_idx, second_idx] += 1
        match_matrix[second_idx, first_idx] += 1
        
        # Increment the win count
        if match['winner'] == match['first']:
            win_matrix[first_idx, second_idx] += 1
        else:
            win_matrix[second_idx, first_idx] += 1

    # Calculate the win probability matrix
    return np.divide(win_matrix, match_matrix, out=np.zeros_like(win_matrix), where=match_matrix!=0)

def compute_judges_win_probability(contestants_index, all_contests:Dict[str,Dict]):
    return {judge: compute_win_probability(contestants_index, contests) for (judge,contests) in all_contests.items()}

# compute win probabilities
judges_probability_matrices = compute_judges_win_probability(contestants_index, contests_data)
print(f"Win Probability matrices: {judges_probability_matrices}")

#--------------------------------------------------------------------------------------------------
# COMPUTE MAJORITY WIN PROBABILITY

def compute_majority_win_probability(win_probability:float, nb_questions_per_trial:int) -> float:
    """
    Calculate the probability that the model wins on more than half of the nb_questions question,
    based on winning win_probability of the time in previous qustions.
    
    Parameters:
    win_probability (float): Estimated probability that the model wins on one question
    nb_questions_per_trial (int): Number of question in a trial
    
    Returns:
    float: Probability that the model wins on more than half of the question.
    """       
    # Sum of binomial probabilities for winning more than nb_questions_per_trial/2 times:
    # stats.binom.pmf(nb_wins, nb_questions_per_trial, win_probability) gives us the probability of winning exactly nb_wins times
    # We sum the binomial probabilities from floor(nb_questions_per_trial/2) + 1 to nb_questions_per_trial
    probability = sum(stats.binom.pmf(nb_wins, nb_questions_per_trial, win_probability) for nb_wins in range(nb_questions_per_trial // 2 + 1, nb_questions_per_trial + 1))
    return probability

def compute_judges_majority_win_probability(judges_probability_matrices, nb_questions):
    vectorized_compute_majority_win_probability = np.vectorize(compute_majority_win_probability)
    return {judge: compute_majority_win_probability(win_probabilities_matrix, nb_questions) for (judge,win_probabilities_matrix) in judges_probability_matrices.items()}

# compute majority win probabilities
judges_majority_win_probability_matrices = compute_judges_majority_win_probability(judges_probability_matrices, nb_questions_per_trial)
print(f"Majority Win Probability matrices: {judges_majority_win_probability_matrices}")

#--------------------------------------------------------------------------------------------------
# MARKOV TRANSITION MATRIX

def compute_markov_transition_matrix(win):
    n = win.shape[0]
    markov = np.zeros_like(win)

    # Compute the non-diagonal elements
    for r in range(n):
        for c in range(n):
            if r != c:
                markov[r, c] = win[r, c] / (n - 1)

    # Compute the diagonal elements
    for r in range(n):
        markov[r, r] = 1 - np.sum(markov[r, :])

    return markov

def compute_judges_markov_matrices(judges_majority_win_probability_matrices):
    return {judge: compute_markov_transition_matrix(win) for (judge,win) in judges_majority_win_probability_matrices.items()}

# compute markov transtion matrices
judges_markov_matrices = compute_judges_markov_matrices(judges_majority_win_probability_matrices)
print(f"Markov transition matrices: {judges_markov_matrices}")

#--------------------------------------------------------------------------------------------------
# MARKOV END PROBABILITIES

def apply_markov_chain(markov_matrix, nb_trials):
    # Initialize the probability vector (equiprobable starting point)
    nb_rows, nb_cols = markov_matrix.shape
    prob_vector = np.ones(nb_rows) / nb_rows
    
    # Apply the Markov matrix nb_trials times
    for _ in range(nb_trials):
        prob_vector = prob_vector @ markov_matrix
    
    # Final renormalization to handle numerical instabilities
    prob_vector /= np.sum(prob_vector)
    return prob_vector

def compute_judges_end_proba(judges_markov_matrices, nb_trials):
    return {judge: apply_markov_chain(markov, nb_trials) for (judge,markov) in judges_markov_matrices.items()}

# compute markov transtion matrices
judges_end_probas = compute_judges_end_proba(judges_markov_matrices, nb_trials)
print(f"End probabilities: {judges_end_probas}")

#--------------------------------------------------------------------------------------------------
# AGREEMENT PROBABILITIES

def agreement_probability(prob_vector1, prob_vector2):
    # Compute the sum of the minimum of paired values from the two probability vectors
    return np.sum(np.minimum(prob_vector1, prob_vector2))

def compute_alignment(models_dict):
    # Extract the human model's probability vector
    human_prob_vector = models_dict.get('human')
    
    # Initialize a dictionary to store the alignment results
    alignment_dict = {}
    
    # Compute the alignment for each model against the human model
    for model, prob_vector in models_dict.items():
        if model != 'human':
            alignment_dict[model] = agreement_probability(prob_vector, human_prob_vector)
    
    return alignment_dict

# compute judges alignements
judges_alignements = compute_alignment(judges_end_probas)
print(f"Alignements: {judges_alignements}")

#--------------------------------------------------------------------------------------------------
# PLOTS

def plot_alignment(alignment_dict, output_file):
    # Sort the alignment dictionary by models and corresponding scores
    sorted_items = sorted(alignment_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Unpack the sorted items into models and alignment scores
    models, alignment_scores = zip(*sorted_items)

    plt.figure(figsize=(12, 8))
    plt.bar(models, alignment_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Alignment with Human')
    plt.title('Model Alignment with Human')
    plt.ylim(0, 1)  # Since alignment is a probability, it should be between 0 and 1
    
    # Tilt the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

def plot_judges_probabilities(judges_end_probas, contestants_index, output_dir):
    # Inverse the contestants_index dictionary to map indices to chatbot names
    index_to_chatbot = {v: k for k, v in contestants_index.items()}
    
    # Iterate over each judge and their corresponding probability vector
    for judge_name, prob_vector in judges_end_probas.items():
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Get the chatbot names sorted by their indices
        chatbot_names = [index_to_chatbot[i] for i in range(len(prob_vector))]
        
        # Plot the probabilities
        plt.bar(chatbot_names, prob_vector, color='skyblue')
        plt.xlabel('Chatbots')
        plt.ylabel('Probability')
        plt.title(f'Probabilities assigned by {judge_name}')
        plt.ylim(0, 1)  # Since it's a probability, it should be between 0 and 1
        
        # Tilt the x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Save each plot to a file named after the judge
        plt.savefig(f'{output_dir}/{judge_name}_probabilities.png')
        plt.close()

# save plots to file
plot_alignment(judges_alignements, alignement_plot_file)
plot_judges_probabilities(judges_end_probas, contestants_index, output_folder)
