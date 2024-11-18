import json
import math
import numpy as np
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

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
    'Pro':'gemini-1.5-pro-api-0514', 
    'Flash':'gemini-1.5-flash-api-0514', 
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

def compute_variance(match_data:Dict) -> Dict:
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

    # Calculate correlation
    correlation, _ = stats.pearsonr(elo_values, metric_values)

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
    plt.plot(elo_values, m * elo_values + b, color="red", label=f"Fit Line (r={correlation:.2f})")

    plt.legend()
    plt.grid()

    # Save plot to file
    filename = f'{metric_name.lower().replace(" ", "_")}.png'
    output_file = output_folder / filename
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

#--------------------------------------------------------------------------------------------------
# MAIN

# Importing the data
elo_data = import_elos(elo_file)
json_data = import_jsons(input_folder)
match_data = dict_map(json_data, lambda data: compute_variance(compute_match(data)))

# Computing the metrics
variance_data = dict_map(match_data, compute_weighted_average_variance)
std_data = dict_map(variance_data, math.sqrt)

# Ploting the correlations
plot_correlation(elo_data, variance_data, metric_name='Variance', output_folder=output_folder)
plot_correlation(elo_data, std_data, metric_name='Standard Deviation', output_folder=output_folder)

"""
TODO:
* try weighting the std more cleverly
  * no weight
  * nb_match weight
  * regression weight
"""