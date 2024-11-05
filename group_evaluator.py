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
input_folder = Path("./data/groups")
human_reference = 'human'

# on how many questions are two models compared in a trial
nb_questions_per_trial = 20

# how many trials are conducted before picking our best, last, winner chatbot model?
# default to number of chatbot models
nb_trials = 10

# output files
output_folder = Path("./outputs/groups")

#--------------------------------------------------------------------------------------------------
# DATA PROCESSING

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

def compute_win_probability(contestants_index, contests, laplacian_smoothing:int):
    """
    laplacian_smoothing is a stabilizing term, added to the number of match to deal with low samples
    (0 would mean no smoothing)
    """
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
    prob_matrix = np.divide(win_matrix, (match_matrix+laplacian_smoothing), out=np.zeros_like(win_matrix), where=match_matrix!=0)
    return match_matrix, prob_matrix

def compute_judges_win_probability(contestants_index, all_contests:Dict[str,Dict], laplacian_smoothing:int=0):
    judges_nbmatch_matrices = dict()
    judges_probability_matrices = dict()
    for (judge,contests) in all_contests.items():
        match_matrix, prob_matrix = compute_win_probability(contestants_index, contests, laplacian_smoothing)
        judges_nbmatch_matrices[judge] = match_matrix
        judges_probability_matrices[judge] = prob_matrix
    return judges_nbmatch_matrices, judges_probability_matrices

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

def compute_alignment(winproba_dict, majoritywin_dict, alignement_function):
    # Extract the human model's probability vector
    if (human_reference in winproba_dict) and (human_reference in majoritywin_dict):
        human_winprob_vector = winproba_dict.get(human_reference)
        human_majwin_mat = majoritywin_dict.get(human_reference)
    else:
        raise RuntimeError(f"There is no '{human_reference}' in our data.")
    
    # Initialize a dictionary to store the alignment results
    alignment_dict = {}
    
    # Compute the alignment for each model against the human model
    for model, judge_majwin_mat in majoritywin_dict.items():
        if model != human_reference:
            judge_winprob_vector = winproba_dict[model]
            alignment_dict[model] = alignement_function(human_winprob_vector, human_majwin_mat, judge_winprob_vector, judge_majwin_mat)
    
    return alignment_dict

def plot_alignment(alignment_dict, output_file:Path):
    # Sort the alignment dictionary by names
    sorted_items = sorted(alignment_dict.items(), key=lambda item: item[0])
    
    # Unpack the sorted items into models and alignment scores
    models, alignment_scores = zip(*sorted_items)

    plt.figure(figsize=(12, 8))
    plt.bar(models, alignment_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(f'Probability of agreeing with Human after {nb_trials} trials of {nb_questions_per_trial} questions')
    plt.title('Model Alignment with Human')
    plt.ylim(0, 1)  # Since alignment is a probability, it should be between 0 and 1
    
    # Tilt the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

def plot_variance(alignment_dict, output_file:Path):
    # Sort the alignment dictionary by names
    sorted_items = sorted(alignment_dict.items(), key=lambda item: item[0])
    
    # Unpack the sorted items into models and alignment scores
    models, alignment_scores = zip(*sorted_items)

    # standardize number
    alignment_scores = (alignment_scores - np.mean(alignment_scores)) / np.std(alignment_scores)

    plt.figure(figsize=(12, 8))
    plt.bar(models, alignment_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(f'Probability of agreeing with Human after {nb_trials} trials of {nb_questions_per_trial} questions')
    plt.title('Variance of Model Alignment with Human')
    plt.ylim(-2.5, 2.5)
    
    # Tilt the x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

#--------------------------------------------------------------------------------------------------
# GROUP PROCESSING

def process_group(input_folder, output_folder, alignement_function):
    """
    Computes the alignement score.
    """
    # does all the computations
    contests_data = import_jsons(input_folder)
    contestants_index = extract_contestants(contests_data)
    judges_nbmatch_matrices, judges_probability_matrices = compute_judges_win_probability(contestants_index, contests_data, laplacian_smoothing=1)
    judges_majority_win_probability_matrices = compute_judges_majority_win_probability(judges_probability_matrices, nb_questions_per_trial)
    judges_markov_matrices = compute_judges_markov_matrices(judges_majority_win_probability_matrices)
    judges_end_probas = compute_judges_end_proba(judges_markov_matrices, nb_trials)
    judges_alignements = compute_alignment(judges_end_probas, judges_majority_win_probability_matrices, alignement_function)
    # plots the result
    plot_alignment(judges_alignements, output_folder / f"alignement_{input_folder.name}.png")
    plot_variance(judges_alignements, output_folder / f"variance_{input_folder.name}.png")

def process_groups(input_folder: Path, output_folder: Path, alignement_function):
    """
    Processes each group folder within the input folder and saves the results to the output folder.

    Args:
        input_folder (Path): Path to the directory containing group folders.
        output_folder (Path): Path to the directory where output should be saved.
    """
    print(f"Processing {output_folder.name}...")
    # Ensure the output folder exists, creating it if necessary
    output_folder.mkdir(parents=True, exist_ok=True)
    # Loop through each subfolder in the input_folder
    for group_input_folder in input_folder.iterdir():
        if group_input_folder.is_dir():  # Ensure it's a directory
            process_group(group_input_folder, output_folder, alignement_function)

#--------------------------------------------------------------------------------------------------
# ALIGNEMENT FUNCTIONS

def winproba(human_winprob_vector, human_majwin_mat, judge_winprob_vector, judge_majwin_mat):
    # Compute the sum of the minimum of paired values from the two probability vectors
    # this is the probability of both distribution having the same value, hypothezing maximum correlation
    return np.sum(np.minimum(human_winprob_vector, judge_winprob_vector)) # min

def weightedpairwiseproba(human_winprob_vector, human_majwin_mat, judge_winprob_vector, judge_majwin_mat):
    """
    compute the proba of the judge and human going the same direction
    over all pairs
    then weight it by the likelyhood of those contests
    """
    nb_chatbots = len(human_winprob_vector)
    # proba of both judge and human saying win, or both saying lose
    proba_agree = (human_majwin_mat * judge_majwin_mat) + ((1.0 - human_majwin_mat) * (1.0 - judge_majwin_mat))
    np.fill_diagonal(proba_agree, 0.0)
    # average it into one column
    # computing the average error associated with each chatbot
    # weighted by the probability of each encounter (equiprobable)
    proba_per_chatbot = np.sum(proba_agree, axis=1) / (nb_chatbots-1)
    # reduce it into a single number
    # weighting the errors by the probability of a chatbot being the current best
    proba = np.dot(human_winprob_vector, proba_per_chatbot)
    return proba

def difficultyweightedpairwiseproba(human_winprob_vector, human_majwin_mat, judge_winprob_vector, judge_majwin_mat):
    """
    compute the proba of the judge and human going the same direction
    over all pairs
    then weight it by the likelyhood of those contests
    """
    nb_chatbots = len(human_winprob_vector)
    # proba of both judge and human saying win, or both saying lose
    proba_agree = (human_majwin_mat * judge_majwin_mat) + ((1.0 - human_majwin_mat) * (1.0 - judge_majwin_mat))
    np.fill_diagonal(proba_agree, 0.0)
    # difficulty of a given pair
    difficulty = 4.0 * human_majwin_mat * (1.0 - human_majwin_mat)
    np.fill_diagonal(difficulty, 0.0)
    difficulty /= (np.sum(difficulty) / (nb_chatbots*(nb_chatbots-1)) )
    # weighting proba agree
    proba = proba_agree * difficulty
    # average it into one column
    # computing the average error associated with each chatbot
    # weighted by the probability of each encounter (equiprobable)
    proba_per_chatbot = np.sum(proba, axis=1) / (nb_chatbots-1)
    # reduce it into a single number
    # weighting the errors by the probability of a chatbot being the current best
    proba = np.dot(human_winprob_vector, proba_per_chatbot)
    return proba

def alignement(human_winprob_vector, human_majwin_mat, judge_winprob_vector, judge_majwin_mat):
    """
    compute the proba of the judge and human going the same direction
    over all pairs
    then weight it by the likelyhood of those contests
    """
    nb_chatbots = len(human_winprob_vector)
    # proba of both judge and human saying win, or both saying lose
    proba_agree = (human_majwin_mat * judge_majwin_mat) + ((1.0 - human_majwin_mat) * (1.0 - judge_majwin_mat))
    np.fill_diagonal(proba_agree, 0.0)
    # difficulty of a given pair
    difficulty = 4.0 * human_majwin_mat * (1.0 - human_majwin_mat)
    np.fill_diagonal(difficulty, 0.0)
    # weighting proba agree
    # a*d + 0.5*(a-1)*d + 0.5*(d-1)*a
    # success on a hard one is great success
    # fail on an easy one is catastrophic fail
    proba = (proba_agree * difficulty) + 0.5*( (1.0-proba_agree)*difficulty + proba_agree*(1.0-difficulty) )
    # average it into one column
    # computing the average error associated with each chatbot
    # weighted by the probability of each encounter (equiprobable)
    proba_per_chatbot = np.sum(proba, axis=1) / (nb_chatbots-1)
    # reduce it into a single number
    # weighting the errors by the probability of a chatbot being the current best
    proba = np.dot(human_winprob_vector, proba_per_chatbot)
    return proba

#--------------------------------------------------------------------------------------------------
# MAIN

# displays the output of the various alignement metrics
process_groups(input_folder, output_folder / 'winproba', winproba)
process_groups(input_folder, output_folder / 'weightedpairwiseproba', weightedpairwiseproba)
process_groups(input_folder, output_folder / 'weightedpairwiseproba', weightedpairwiseproba)
process_groups(input_folder, output_folder / 'difficultyweightedpairwiseproba', difficultyweightedpairwiseproba)
process_groups(input_folder, output_folder / 'alignement', alignement)
print("Done.")
