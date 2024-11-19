import json
import math
import numpy as np
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import lsq_linear

# folder with the input jsons
input_folder = Path("./data/merged_groups")
elo_file = input_folder / '..' / 'elo_ratings.json'
human_reference = 'human'

# folder with the output data
output_folder = Path("./outputs/merged_groups")

#--------------------------------------------------------------------------------------------------
# DATA IMPORTATION

def import_json(input_file:Path):
    """
    Reads a single json
    """
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

def import_jsons(input_folder:Path) -> Dict:
    """
    Produces a dict of judge -> judge data
    """
    # Initialize an empty dictionary to hold the data
    data_dict = {}

    # Iterate over all json files in the input folder
    for json_file in input_folder.glob('*_contest_data.json'):
        # Extract the judge name part of the filename
        name = json_file.stem.split('_contest_data')[0]
        
        # Add the data to the dictionary with the judge name as the key
        data_dict[name] = import_json(json_file)
    
    return data_dict

def import_elos(input_file:Path) -> Dict:
    return {kv['Model']:kv['Elo rating'] for kv in import_json(input_file)}

# table converting from our short names to official ones
shortname_conversion = {
    'Starling':'starling-lm-7b-beta',
    'GPT-4o-mini':'gpt-4o-mini-2024-07-18', 
    'Mistral-Large':'mistral-large-2407',
    'Command-R+': 'command-r-plus', 
    'human':'human', 
    'Llama-3.1-8b':'llama-3.1-8b-instruct', 
    'Haiku':'claude-3-haiku-20240307', 
    'Pro':'gemini-1.5-pro-002', 
    'Flash':'gemini-1.5-flash-002', 
    'Command-R':'command-r', 
    'Llama-3.1-405b':'llama-3.1-405b-instruct'
}

#--------------------------------------------------------------------------------------------------
# PER-JUDGE DATA-PROCESSING

def dict_map(input_dict:Dict, f) -> Dict:
    """
    applies a function to all values in a dict
    returns the new, modified, dict
    """
    return { key:f(value) for (key,value) in input_dict.items() }

def compute_match(contest_data:List[Dict]) -> Dict:
    """
    Takes a list of contests
    produces a dictionnary of (model1,model2) -> (nb_matches,nb_win1)
    """
    result = defaultdict(lambda: {'nb_matches': 0, 'nb_wins1': 0.0})
    for contest in contest_data:
        # pair in alphabetical order
        pair = (contest['first'], contest['second']) if (contest['first'] < contest['second']) else (contest['second'], contest['first'])
        # increment number of matches
        result[pair]['nb_matches'] += 1
        # increment number of wins
        if pair[0] == contest['winner']:
            # pair's first mode won
            result[pair]['nb_wins1'] += 1
        elif pair[1] != contest['winner']:
            # tie
            result[pair]['nb_wins1'] += 0.5
    return result

def add_variance(match_data:Dict) -> Dict:
    """
    takes a dictionnary of (model1,model2) -> (nb_matches,nb_win1)
    adds win proba and variance fields to each subdict
    """
    for pair_match_data in match_data.values():
        win_proba = pair_match_data['nb_wins1'] / pair_match_data['nb_matches']
        variance = win_proba * (1.0 - win_proba)
        pair_match_data['win_proba'] = win_proba
        pair_match_data['variance'] = variance
    return match_data

def add_elodiff(match_data:Dict, elo_scores:Dict) -> Dict:
    """
    takes a dictionnary of (model1,model2) -> (nb_matches,nb_win1)
    adds elo_diff to each pair
    """
    for (pair, pair_match_data) in match_data.items():
        elo0 = elo_scores[pair[0]]
        elo1 = elo_scores[pair[1]]
        pair_match_data['elo_diff'] = abs(elo0 - elo1)
    return match_data

#--------------------------------------------------------------------------------------------------
# METRICS

def compute_average_variance(match_data:Dict) -> float:
    """
    takes a dictionnary of (model1,model2) -> (nb_matches,nb_win1,win_proba,variance)
    returns the average variance
    """
    return np.mean( [data['variance'] for data in match_data.values()] )

def compute_weighted_average_variance(match_data:Dict) -> float:
    """
    takes a dictionnary of (model1,model2) -> (nb_matches,nb_win1,win_proba,variance)
    returns the average variance, weighted by nb_matches
    """
    total_variance = 0.0
    total_weight = 0
    for pair_match_data in match_data.values():
        weight = pair_match_data['nb_matches']
        total_variance += pair_match_data['variance'] * weight
        total_weight += weight 
    return total_variance / total_weight

def compute_pairweighted_variance(match_data:Dict, pair_weights:Dict) -> float:
    """
    compute vairance, weighted with precomputed pair weights
    """
    total_variance = 0.0
    total_weight = 0
    for pair, pair_match_data in match_data.items():
        weight = pair_weights[pair]
        total_weight += weight
        total_variance += pair_match_data['variance'] * weight
    return total_variance / total_weight

#--------------------------------------------------------------------------------------------------
# REGRESSION

def regress_pair_weights(match_data:Dict, elo_scores:Dict, solver='nonneg') -> Dict:
    """
    produces weights for pairs, such that sum(var*weights) = elo for the judges
    """
    judges_names = list(match_data.keys())
    # get the names of the pairs common to all judges
    pairs = None
    for judge_data in match_data.values():
        judge_pairs = judge_data.keys()
        if (pairs is None):
            pairs = set(judge_pairs)
        else:
            pairs.intersection_update(judge_pairs)
    pairs = list(pairs)
    # build a matrix where each row is a judge, and each column is a pair (in naming order)
    variance_matrix = np.array([ [ match_data[judge][pair]['variance'] for pair in pairs] for judge in judges_names ])
    # build a vector of elo scores
    elo_vector = np.array([ elo_scores[shortname_conversion[judge]] for judge in judges_names ])
    # do the linear regresssion to find weights for the pairs
    if solver == 'lstsq':
        weights, residuals, rank, singular_values = np.linalg.lstsq(variance_matrix, elo_vector, rcond=None)
    elif solver == 'nonneg':
        result = lsq_linear(variance_matrix, elo_vector, bounds=(0, np.inf))
        weights = result.x
    # save it as a dict of pair->weight
    return { pair:weight for (pair,weight) in zip(pairs,weights) }

#--------------------------------------------------------------------------------------------------
# PLOTTING

def plot_correlation(elo_data: Dict, metric_data: Dict, metric_name: str, output_folder: Path):
    """
    Create and save a plot of ELO vs metric with a correlation line and model annotations.

    Parameters:
    - elo_data (dict): Dictionary mapping official names to ELO ratings.
    - metric_data (dict): Dictionary mapping short names to metric values.
    - metric_name (str): Name of the metric for labeling.
    - output_folder (Path): Path to save the plot.
    - shortname_conversion (dict): Mapping from short names to official names.
    - human_reference (str): Short name for the human reference to ignore.
    """
    # Find common keys
    judge_names = metric_data.keys() - {human_reference}  # Ignoring human reference
    elo_values = np.array([elo_data[shortname_conversion[key]] for key in judge_names])
    metric_values = np.array([metric_data[key] for key in judge_names])

    # Calculate correlations
    pearson_corr, _ = stats.pearsonr(elo_values, metric_values)
    spearman_corr, _ = stats.spearmanr(elo_values, metric_values)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(elo_values, metric_values, label="Data Points")
    plt.xlabel("ELO Rating")
    plt.ylabel(metric_name)
    plt.title(f"ELO vs {metric_name}")

    # Annotate each point with the short name
    for x, y, name in zip(elo_values, metric_values, judge_names):
        plt.text(x, y, name, fontsize=9, ha='right', va='bottom', alpha=0.7)

    # Add correlation line
    m, b = np.polyfit(elo_values, metric_values, 1)
    plt.plot(elo_values, m * elo_values + b, color="red", 
             label=f"Fit Line (Pearson r={pearson_corr:.2f}, Spearman ρ={spearman_corr:.2f})")

    plt.legend()
    plt.grid()

    # Save plot to file
    filename = f'{metric_name.lower().replace(" ", "_")}.png'
    output_file = output_folder / filename
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

def plot_weight_correlation(elo_data: Dict, pair_weights:Dict, output_folder: Path):
    pair_names = list(pair_weights.keys())
    weight_values = np.array([pair_weights[p] for p in pair_names])
    elodiff_values = np.array([ abs(elo_data[p[0]] - elo_data[p[1]]) for p in pair_names ])

    # Calculate correlation
    correlation, _ = stats.pearsonr(elodiff_values, weight_values)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(elodiff_values, weight_values, label="Data Points")
    plt.xlabel("ELO Difference")
    plt.ylabel("Pair Weights")
    plt.title(f"Pair Weights vs ELO Differences")

    # Add correlation line
    m, b = np.polyfit(elodiff_values, weight_values, 1)
    plt.plot(elodiff_values, m * elodiff_values + b, color="red", label=f"Fit Line (r={correlation:.2f})")

    plt.legend()
    plt.grid()

    # Save plot to file
    output_file = output_folder / 'pair_weights_vs_elo_difference.png'
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

#--------------------------------------------------------------------------------------------------
# MAIN

# Importing the data
elo_data = import_elos(elo_file)
json_data = import_jsons(input_folder)
match_data = dict_map(json_data, lambda data: add_variance(compute_match(data)) )
nonhuman_match_data = {k: v for k, v in match_data.items() if k != human_reference}

# Run Regression
regression_pair_weights = regress_pair_weights(nonhuman_match_data, elo_data)
plot_weight_correlation(elo_data, regression_pair_weights, output_folder)
pair_elodiff = { pair:abs(elo_data[pair[0]] - elo_data[pair[1]]) for pair in regression_pair_weights.keys() }
exp_pair_weights = { pair: np.exp(-elodiff / 100) for (pair,elodiff) in pair_elodiff.items() }
log_pair_weights = { pair: np.log10(1 + elodiff) for (pair,elodiff) in pair_elodiff.items() }

# Computing the metrics
variance_data = dict_map(match_data, compute_average_variance)
weighted_variance_data = dict_map(match_data, compute_weighted_average_variance)
regression_weighted_variance_data = dict_map(match_data, lambda data: compute_pairweighted_variance(data,regression_pair_weights))
expelodiff_weighted_variance_data = dict_map(match_data, lambda data: compute_pairweighted_variance(data,exp_pair_weights))
logelodiff_weighted_variance_data = dict_map(match_data, lambda data: compute_pairweighted_variance(data,log_pair_weights))

# Ploting the correlations
plot_correlation(elo_data, variance_data, metric_name='Variance', output_folder=output_folder)
plot_correlation(elo_data, weighted_variance_data, metric_name='Weighted Variance', output_folder=output_folder)
plot_correlation(elo_data, regression_weighted_variance_data, metric_name='Regression Pair Weighted Variance', output_folder=output_folder)
plot_correlation(elo_data, expelodiff_weighted_variance_data, metric_name='Exp ELodiff Pair Weighted Variance', output_folder=output_folder)
plot_correlation(elo_data, logelodiff_weighted_variance_data, metric_name='Log ELodiff Pair Weighted Variance', output_folder=output_folder)

"""
Results:
* sd vs variance give the same numbers correlation, might as well use variance
* weighted variance vs variance: essentially the same, in part because we have roughly equal amounts of data everywhere
* basic (and non-negative) regression overfits on the number of pairs
* spearman correlation is more discriminant here
"""