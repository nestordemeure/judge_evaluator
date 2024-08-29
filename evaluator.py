import json
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict
from pathlib import Path

# folder with the input jsons
input_folder = Path("./data")
human_reference = 'human_all'

# on how many questions are two models compared in a trial
nb_questions_per_trial = 20

# how many trials are conducted before picking our best, last, winner chatbot model?
# default to number of chatbot models
nb_trials = 10

# output files
output_folder = Path("./outputs")
alignement_plot_file = output_folder / "judges_alignment.png"

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
print("Loading Data...")
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
print("Running Comptations...")
contestants_index = extract_contestants(contests_data)

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

#--------------------------------------------------------------------------------------------------
# COMPUTE MAJORITY WIN PROBABILITY

def compute_majority_win_probability(win_probability:float, nb_questions_per_trial:int) -> float:
    """
    Calculate the probability that the model wins on more than half of the nb_questions question,
    based on winning win_probability of the time in previous qustions.

    :param win_probability: Estimated probability that the model wins on one question
    :param nb_questions_per_trial: Number of question in a trial
    :return: Probability that the model wins on more than half of the question.
    """
    # "Most of the time" means more than half the time
    k = math.ceil(nb_questions_per_trial / 2)
    
    # Calculate the cumulative probability
    #cumulative_prob = 0.0
    #for i in range(k, nb_questions_per_trial + 1):
    #    # Calculate binomial coefficient
    #    binomial_coeff = math.comb(nb_questions_per_trial, i) 
    #    # Calculate probability for exactly i successes
    #    prob = binomial_coeff * (win_probability ** i) * ((1 - win_probability) ** (nb_questions_per_trial - i))
    #    # Add to cumulative probability
    #    cumulative_prob += prob
    
    # Calculate the cumulative probability using scipy.stats.binom.sf
    # sf gives P(X >= k), which is what we want
    cumulative_prob = stats.binom.sf(k - 1, nb_questions_per_trial, win_probability)

    return cumulative_prob

def compute_judges_majority_win_probability(judges_probability_matrices, nb_questions):
    return {judge: compute_majority_win_probability(win_probabilities_matrix, nb_questions) for (judge,win_probabilities_matrix) in judges_probability_matrices.items()}

# compute majority win probabilities
judges_majority_win_probability_matrices = compute_judges_majority_win_probability(judges_probability_matrices, nb_questions_per_trial)

#--------------------------------------------------------------------------------------------------
# MARKOV TRANSITION MATRIX

def compute_markov_transition_matrix(win):
    markov = np.zeros_like(win)

    # Compute the non-diagonal elements
    n = win.shape[0]
    for r in range(n):
        for c in range(n):
            if r != c:
                # NOTE: transition probability is the probability of us (r) losing against them (c) (1 - win[r,c])
                #       times probability of a trial between the two ( 1 / (n-1) )
                proba_losing = 1.0 - win[r, c]
                markov[r, c] = proba_losing / (n - 1)

    # Compute the diagonal elements
    # probability of staying where we are
    for r in range(n):
        markov[r, r] = 1 - np.sum(markov[r, :])

    return markov

def compute_judges_markov_matrices(judges_majority_win_probability_matrices):
    return {judge: compute_markov_transition_matrix(win) for (judge,win) in judges_majority_win_probability_matrices.items()}

# compute markov transtion matrices
judges_markov_matrices = compute_judges_markov_matrices(judges_majority_win_probability_matrices)

#--------------------------------------------------------------------------------------------------
# MARKOV END PROBABILITIES

def apply_markov_chain(markov_matrix, nb_trials):
    # Initialize the probability vector (equiprobable starting point)
    nb_rows, nb_cols = markov_matrix.shape
    prob_vector = np.ones(nb_rows) / nb_rows
    
    # Apply the Markov matrix nb_trials times
    #for _ in range(nb_trials):
    #    prob_vector = prob_vector @ markov_matrix
    # Matrix exponentiation for large numbers of trials
    prob_vector = prob_vector @ np.linalg.matrix_power(markov_matrix, nb_trials)
    
    # Final renormalization to handle potential numerical instabilities
    prob_vector /= np.sum(prob_vector)
    return prob_vector

def compute_judges_end_proba(judges_markov_matrices, nb_trials):
    return {judge: apply_markov_chain(markov, nb_trials) for (judge,markov) in judges_markov_matrices.items()}

# compute markov transtion matrices
judges_end_probas = compute_judges_end_proba(judges_markov_matrices, nb_trials)

#--------------------------------------------------------------------------------------------------
# AGREEMENT PROBABILITIES

def agreement_probability(prob_vector1, prob_vector2):
    # Compute the sum of the minimum of paired values from the two probability vectors
    # this is the probability of both distribution having the same value, hypothezing maximum correlation
    return np.sum(np.minimum(prob_vector1, prob_vector2)) # min

def compute_alignment(models_dict):
    # Extract the human model's probability vector
    human_prob_vector = models_dict.get(human_reference)
    
    # Initialize a dictionary to store the alignment results
    alignment_dict = {}
    
    # Compute the alignment for each model against the human model
    for model, prob_vector in models_dict.items():
        if model != human_reference:
            alignment_dict[model] = agreement_probability(prob_vector, human_prob_vector)
    
    return alignment_dict

# compute judges alignements
judges_alignements = compute_alignment(judges_end_probas)

#--------------------------------------------------------------------------------------------------
# PLOTS

def plot_heatmaps(judges_dict, contestants_index, output_folder):
    """
    Generates and saves a heatmap for each judge's probability matrix.
    
    Parameters:
    judges_dict (dict): A dictionary where keys are judge names (str) and values are numpy matrices (2D arrays of probabilities).
    contestants_index (dict): A dictionary where keys are model names (str) and values are the index (int) corresponding to row/col in the matrix.
    output_folder (Path): The path to the output folder where heatmaps should be saved.
    """
    # Convert the contestants_index dictionary to a list for ordering labels
    index_to_label = {v: k for k, v in contestants_index.items()}
    labels = [index_to_label[i] for i in range(len(index_to_label))]

    for judge, matrix in judges_dict.items():
        plt.figure(figsize=(8, 8))
        
        # Generate the heatmap with color scale from 0 to 0.6
        #sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=labels, yticklabels=labels, vmin=0, vmax=0.6)
        
        # Apply a logarithmic transformation for the color scale
        transformed_matrix = np.log1p(1000 * matrix)  # log1p is used to avoid log(0) by computing log(1 + matrix)
        vmax = np.log1p(1000 * 1.0)

        # Generate the heatmap with the transformed matrix for color scale but original values for annotation
        sns.heatmap(transformed_matrix, annot=matrix, fmt=".2f", cmap='coolwarm', xticklabels=labels, yticklabels=labels, cbar=False, vmin=0, vmax=vmax)
        
        # Add title and labels
        plt.title(f'Probabilities for Judge: {judge}')
        
        # Save the heatmap to the specified folder with the judge's name
        plt.savefig(output_folder / f'{judge}.png')
        
        # Close the plot to free up memory
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
        plt.ylabel(f'Probability of picking a given model after {nb_trials} trials of {nb_questions_per_trial} questions')
        plt.title(f'Probability Distribution of {judge_name}')
        plt.ylim(0, 1)  # Since it's a probability, it should be between 0 and 1
        
        # Tilt the x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Save each plot to a file named after the judge
        plt.savefig(f'{output_dir}/{judge_name}_probabilities.png')
        plt.close()

def plot_pca(judges_end_probas, output_folder):
    # Convert dictionary values to a 2D matrix
    prob_matrix = np.array(list(judges_end_probas.values()))
    names = list(judges_end_probas.keys())

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(prob_matrix)
    explained_variance = pca.explained_variance_ratio_[:2].sum()

    # Plot the PCA result
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])

    # Annotate points with names
    for i, name in enumerate(names):
        plt.annotate(name, (pca_result[i, 0], pca_result[i, 1]))

    plt.title(f"PCA of Judges' Probability Distributions ({explained_variance:.2f} explained variance)")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(output_folder /'judges_pca.png')
    plt.close()

def plot_alignment(alignment_dict, output_file):
    # Sort the alignment dictionary by models and corresponding scores
    sorted_items = sorted(alignment_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Unpack the sorted items into models and alignment scores
    models, alignment_scores = zip(*sorted_items)

    plt.figure(figsize=(12, 8))
    plt.bar(models, alignment_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(f'Probability of agreeing with Human after {nb_trials} trials of {nb_questions_per_trial} questions')
    plt.title('Model Alignment with Human')
    #plt.ylim(0, 1)  # Since alignment is a probability, it should be between 0 and 1
    
    # Tilt the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

# save plots to file
print("Plotting results...")
# matrices
plot_heatmaps(judges_probability_matrices, contestants_index, output_folder / 'pairwise win proba')
plot_heatmaps(judges_majority_win_probability_matrices, contestants_index, output_folder / 'pairwise trial win proba')
plot_heatmaps(judges_markov_matrices, contestants_index, output_folder / 'markov transition matrices')
# winner distrinbutions
plot_judges_probabilities(judges_end_probas, contestants_index, output_folder / 'win distribution')
plot_alignment(judges_alignements, alignement_plot_file)
plot_pca(judges_end_probas, output_folder)
