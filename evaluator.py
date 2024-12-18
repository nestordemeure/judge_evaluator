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
input_folder = Path("./data/nersc")
human_reference = 'human'

# on how many questions are two models compared in a trial
nb_questions_per_trial = 20

# how many trials are conducted before picking our best, last, winner chatbot model?
# default to number of chatbot models
nb_trials = 10

# output files
output_folder = Path("./outputs/nersc")
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
        elif match['winner'] == match['second']:
            win_matrix[second_idx, first_idx] += 1
        else:
            win_matrix[first_idx, second_idx] += 0.5
            win_matrix[second_idx, first_idx] += 0.5

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

# compute win probabilities
judges_nbmatch_matrices, judges_probability_matrices = compute_judges_win_probability(contestants_index, contests_data, laplacian_smoothing=1)

#--------------------------------------------------------------------------------------------------
# Bradley-Terry MODEL

def compute_bradley_terry(match_matrix, probs_matrix, max_iter=100, tol=1e-4):
    """
    Computes the Bradley-Terry probabilities using the iterative method.

    Parameters:
    - match_matrix: A 2D numpy array where [i, j] contains the number of matches between model i and model j.
    - probs_matrix: A 2D numpy array where [i, j] contains the probability that model i won against model j.
    - max_iter: Maximum number of iterations for convergence.
    - tol: Tolerance for convergence.

    Returns:
    - prop: A 2D numpy array where [i, j] contains the probability that model i wins against model j.
    """
    # Computes Bradley-Terry scores iteratively
    n_models = match_matrix.shape[0]  # Number of models
    ratings = np.ones(n_models)  # Initialize all ratings to 1
    wins_matrix = probs_matrix * match_matrix
    for iteration in range(max_iter):
        # updates ratings
        new_ratings = np.zeros_like(ratings)
        for i in range(n_models):
            numerator = np.sum(wins_matrix[i,:] * ratings / (ratings[i] + ratings))
            denominator = np.sum(wins_matrix[:,i] / (ratings[i] + ratings))
            new_ratings[i] = numerator / denominator

        # Normalize ratings by their geometric mean
        new_ratings /= stats.gmean(new_ratings + 1e-10)  # Avoid log(0) by adding a small value
        # Check for convergence
        if np.max(np.abs(new_ratings - ratings)) < tol:
            break

        ratings = new_ratings

    # Compute the probabilities
    prop = np.zeros_like(match_matrix, dtype=float)
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                prop[i, j] = ratings[i] / (ratings[i] + ratings[j])
    return prop

def compute_judges_bradley_terry(judges_match_matrices, judges_prob_matrices, max_iter=100, tol=1e-4):
    result = dict()
    for judge, match_matrix in judges_match_matrices.items():
        probs_matrix = judges_prob_matrices[judge]
        bt_matrix = compute_bradley_terry(match_matrix, probs_matrix, max_iter, tol)
        result[judge] = bt_matrix
    return result

# uses bradley terry to smooth probabilities
# on ALL models
# judges_probability_matrices = compute_judges_bradley_terry(judges_nbmatch_matrices, judges_probability_matrices)
# on the human ONLY
# judges_probability_matrices[human_reference] = compute_bradley_terry(judges_nbmatch_matrices[human_reference], judges_probability_matrices[human_reference])

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
    if human_reference in models_dict:
        human_prob_vector = models_dict.get(human_reference)
    else:
        raise RuntimeError(f"There is no '{human_reference}' in our data.")
    
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
# AGREEMENT PROBABILITIES

def weighted_error_alignement(human_winprob_vector, human_majwin_mat, judge_majwin_mat):
    """
    pairwise win proba error, weighted by the probability of each pairing
    
    The idea is to look at the probability of any pairing of chatbot: proba that a model is the current best * (equiprobable) proba that it faces another model
    Then, sum the error on each pairwise probility (human vs current judge) weighted by the above proba of that pairing.

    This lets us consider the error we make at the pairwise level, but not give undue attention to unlikely pairs of bad models.
    The end result is garanteed to be in [0;1]

    Note: we could try and go one step further, replacing the equiprobable proba with the win proba of the other model
    """
    nb_chatbots = len(human_winprob_vector)
    # compute L1 error of pairwise majority win proba
    l1_error = np.abs(human_majwin_mat - judge_majwin_mat)
    np.fill_diagonal(l1_error, 0.0)
    # average it into one column
    # computing the average error associated with each chatbot
    # weighted by the probability of each encounter (equiprobable)
    error_per_chatbot = np.sum(l1_error, axis=1) / (nb_chatbots-1)
    # reduce it into a single number
    # weighting the errors by the probability of a chatbot being the current best
    error = np.dot(human_winprob_vector, error_per_chatbot)
    return 1 - error

def compute_alignment2(winproba_dict, majoritywin_dict):
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
            alignment_dict[model] = weighted_error_alignement(human_winprob_vector, human_majwin_mat, judge_majwin_mat)
    
    return alignment_dict

# compute judges alignements
judges_alignements2 = compute_alignment2(judges_end_probas, judges_majority_win_probability_matrices)

#--------------------------------------------------------------------------------------------------
# AGREEMENT PROBABILITIES

def weighted_error_alignement2(human_winprob_vector, human_majwin_mat, judge_majwin_mat):
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

def compute_alignment3(winproba_dict, majoritywin_dict):
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
            alignment_dict[model] = weighted_error_alignement2(human_winprob_vector, human_majwin_mat, judge_majwin_mat)
    
    return alignment_dict

# compute judges alignements
judges_alignements3 = compute_alignment3(judges_end_probas, judges_majority_win_probability_matrices)

#--------------------------------------------------------------------------------------------------
# AGREEMENT PROBABILITIES

def weighted_error_alignement4(human_winprob_vector, human_majwin_mat, judge_majwin_mat):
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

def compute_alignment4(winproba_dict, majoritywin_dict):
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
            alignment_dict[model] = weighted_error_alignement4(human_winprob_vector, human_majwin_mat, judge_majwin_mat)
    
    return alignment_dict

# compute judges alignements
judges_alignements4 = compute_alignment4(judges_end_probas, judges_majority_win_probability_matrices)


#--------------------------------------------------------------------------------------------------
# PLOTS

def plot_integermatrix(judges_dict, contestants_index, output_folder):
    """
    Generates and saves a heatmap for each judge's probability matrix.
    
    Parameters:
    judges_dict (dict): A dictionary where keys are judge names (str) and values are numpy matrices (2D arrays of probabilities).
    contestants_index (dict): A dictionary where keys are model names (str) and values are the index (int) corresponding to row/col in the matrix.
    output_folder (Path): The path to the output folder where heatmaps should be saved.
    """
    # ensures the folder exists
    output_folder.mkdir(exist_ok=True)
    # Convert the contestants_index dictionary to a list for ordering labels
    index_to_label = {v: k for k, v in contestants_index.items()}
    labels = [index_to_label[i] for i in range(len(index_to_label))]

    # maximum value accross all matrices, to keep colors consistant
    nb_max = max( np.max(matrix) for matrix in judges_dict.values() )

    for judge, matrix in judges_dict.items():
        plt.figure(figsize=(8, 8))

        # Generate the heatmap with the transformed matrix for color scale but original values for annotation
        sns.heatmap(matrix, annot=matrix, cmap='coolwarm', xticklabels=labels, yticklabels=labels, cbar=False, vmin=0, vmax=nb_max)
        
        # Add title and labels
        plt.title(f'Number of Matches for Judge: {judge}')
        
        # Save the heatmap to the specified folder with the judge's name
        plt.savefig(output_folder / f'{judge}.png')
        
        # Close the plot to free up memory
        plt.close()

def plot_heatmaps(judges_dict, contestants_index, output_folder):
    """
    Generates and saves a heatmap for each judge's probability matrix.
    
    Parameters:
    judges_dict (dict): A dictionary where keys are judge names (str) and values are numpy matrices (2D arrays of probabilities).
    contestants_index (dict): A dictionary where keys are model names (str) and values are the index (int) corresponding to row/col in the matrix.
    output_folder (Path): The path to the output folder where heatmaps should be saved.
    """
    # ensures the folder exists
    output_folder.mkdir(exist_ok=True)
    # Convert the contestants_index dictionary to a list for ordering labels
    index_to_label = {v: k for k, v in contestants_index.items()}
    labels = [index_to_label[i] for i in range(len(index_to_label))]

    for judge, matrix in judges_dict.items():
        plt.figure(figsize=(8, 8))
        
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
    # ensures the folder exists
    output_dir.mkdir(exist_ok=True)
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
    # ensures the folder exists
    output_folder.mkdir(exist_ok=True)
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
plot_integermatrix(judges_nbmatch_matrices, contestants_index, output_folder / 'pairwise nb matches')
plot_heatmaps(judges_probability_matrices, contestants_index, output_folder / 'pairwise win proba')
plot_heatmaps(judges_majority_win_probability_matrices, contestants_index, output_folder / 'pairwise trial win proba')
plot_heatmaps(judges_markov_matrices, contestants_index, output_folder / 'markov transition matrices')
# winner distributions
plot_judges_probabilities(judges_end_probas, contestants_index, output_folder / 'win distribution')
plot_alignment(judges_alignements, alignement_plot_file)
plot_alignment(judges_alignements2, output_folder / "judges_alignment_weightedpairwiseerror.png")
plot_alignment(judges_alignements3, output_folder / "judges_alignment_weightedpairwiseproba.png")
plot_alignment(judges_alignements4, output_folder / "judges_alignment_difficultyweightedpairwiseproba.png")
plot_pca(judges_end_probas, output_folder)
