import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import grad, hessian

# folder with the input jsons
input_folder = Path("./data")
human_reference = 'human'

# output files
output_folder = Path("./outputs")

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
human_data = contests_data[human_reference]

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
num_contestants = len(contestants_index)

#--------------------------------------------------------------------------------------------------
# SET-UP training data

def extract_training_data(contest_data, contestants_index=contestants_index):
    first_idxs = []
    second_idxs = []
    questions = []
    labels = []
    for match in contest_data:
        first_idxs.append(contestants_index[match["first"]])
        second_idxs.append(contestants_index[match["second"]])
        questions.append(match["question"])
        labels.append(1 if (match["winner"] == match["first"]) else 0)
    return first_idxs, second_idxs, questions, labels

first_idxs, second_idxs, questions, labels = extract_training_data(human_data, contestants_index)
num_questions = len(set(questions))
first_idxs = jnp.array(first_idxs)
second_idxs = jnp.array(second_idxs)
questions = jnp.array(questions)
labels = jnp.array(labels)

#--------------------------------------------------------------------------------------------------
# PROBABILITIES

def prob_win_most(p):
    nb_questions = len(p)
    # compute the probability of winning exactly each possible number of wins
    dp0 = jnp.concatenate([jnp.array([1.0]), jnp.zeros(nb_questions)])  # Initial dp array
    def update(dp, proba_winning_question):
        # Shift dp right by one position and multiply by pi (probability of win) / moving up one win
        dp_win = jnp.concatenate([jnp.array([0.0]), dp[:-1]]) * proba_winning_question
        # Multiply dp by (1 - pi) (probability of loss / staying where we are)
        dp_lose = dp * (1 - proba_winning_question)
        # Combine to get new dp
        dp_new = dp_win + dp_lose
        return dp_new, None  # Second value is unused but required by scan
    dp_final, _ = jax.lax.scan(update, dp0, p)

    # sums last elements to compute the proba of winning most questions
    most_questions = nb_questions // 2 + 1  # Calculate the majority threshold
    return dp_final[most_questions:].sum()
# Vectorize the function over the first two dimensions
prob_win_most_matrix = jax.vmap(jax.vmap(prob_win_most, in_axes=0), in_axes=0)

# Define the function to compute the win probabilities
@jax.jit
def compute_win_probabilities(weights_tuple):
    embeddings, weight = weights_tuple
    num_contestants = embeddings.shape[0]
    num_questions = weight.shape[0]  # Get the number of questions
    
    # Define a function to compute the win probability for a single pair (i, j) and a single question
    def win_probability(emb_i, emb_j, question_emb):
        # Compute the difference in embeddings
        emb_diff = emb_i - emb_j
        # Compute the logits by taking the dot product of the question embedding with the contestant embedding difference
        logits = jnp.dot(question_emb, emb_diff)
        # Apply the sigmoid to get the probability that i wins against j for this question
        return jax.nn.sigmoid(logits)
    
    # Vectorize over the questions to get probabilities for all questions
    vectorized_win_probability_q = jax.vmap(win_probability, in_axes=(None, None, 0))
    # Vectorize over the second contestant (j axis)
    vectorized_win_probability_j = jax.vmap(vectorized_win_probability_q, in_axes=(None, 0, None))
    # Vectorize over the first contestant (i axis)
    vectorized_win_probability_ij = jax.vmap(vectorized_win_probability_j, in_axes=(0, None, None))
    
    # Compute probabilities for all (i, j) pairs and sum over all questions
    probabilities = vectorized_win_probability_ij(embeddings, embeddings, weight)
    
    # Sum over the probabilities for all questions
    #win_proba = jnp.mean(probabilities, axis=-1)  # average over the question axis
    win_proba = prob_win_most_matrix(probabilities)  # proba of winning most questions
    #win_proba = jnp.power(jnp.prod(probabilities, axis=-1), 1.0 / num_questions) # geometric mean, gets 0ish everywhere
    return win_proba

#--------------------------------------------------------------------------------------------------
# first order optimization

# Initialize embeddings randomly
key = jax.random.PRNGKey(0)
embedding_size = min(600, round(1.6 * num_contestants ** .56)) # rule of thumb, see https://forums.fast.ai/t/embedding-layer-size-rule/50691
embeddings = jax.random.truncated_normal(key, lower=-1, upper=1, shape=(num_contestants, embedding_size)) / jnp.sqrt(num_contestants) # tensorflow-like embeddings initialization
weight = jnp.ones((num_questions, embedding_size)) / embedding_size
weights_tuple = (embeddings, weight)  # Both embeddings and weight are now being updated

# Sigmoid binary cross-entropy loss function (as defined in optax)
def sigmoid_binary_cross_entropy(logits, labels):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p

# Loss function: computes logits based on the difference of embeddings, applies the question embeddings
def loss_fn(weights_tuple, first_idxs, second_idxs, questions, labels):
    embeddings, weight = weights_tuple
    emb_first = embeddings[first_idxs]
    emb_second = embeddings[second_idxs]
    question_emb = weight[questions]
    
    # Compute logits (dot product of question embedding with the difference in embeddings)
    logits = jnp.sum(question_emb * (emb_first - emb_second), axis=1)
    
    # Compute the loss
    losses = sigmoid_binary_cross_entropy(logits, labels)
    return jnp.mean(losses)

# Gradient descent update function (JIT-compiled)
def update(weights_tuple, first_idxs, second_idxs, questions, labels, learning_rate):
    # Compute the loss and its gradient w.r.t. the parameters
    loss, grads = jax.value_and_grad(loss_fn)(weights_tuple, first_idxs, second_idxs, questions, labels)
    
    # Gradient descent update
    new_embeddings = weights_tuple[0] - learning_rate * grads[0]
    new_weight = weights_tuple[1] - learning_rate * grads[1]
    
    return (new_embeddings, new_weight), loss
update_jitted = jax.jit(fun=update, static_argnames='learning_rate', donate_argnames='weights_tuple')

# Training loop
learning_rate = 1e-3
num_epochs = 10000
for epoch in range(num_epochs):
    # Update the weights tuple (embeddings and weight matrix) using the jitted update function
    weights_tuple, loss = update_jitted(weights_tuple, first_idxs, second_idxs, questions, labels, learning_rate)
    
    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Use final_embeddings for probability calculations
first_order_weights_tuple = weights_tuple
first_order_probabilities = compute_win_probabilities(first_order_weights_tuple)

#--------------------------------------------------------------------------------------------------
# second order optimization

# reinit embeddings
embeddings = jax.random.truncated_normal(key, lower=-1, upper=1, shape=(num_contestants, embedding_size)) / jnp.sqrt(num_contestants) # tensorflow-like embeddings initialization
weight = jnp.ones((num_questions, embedding_size)) / embedding_size  # Initialize weight as a matrix for questions
weights_tuple = (embeddings, weight)  # Both embeddings and weight are now being updated

# Newton's method update function (JIT-compiled)
def update(weights_tuple, first_idxs, second_idxs, questions, labels, learning_rate):
    # Compute the loss and the gradient w.r.t. the parameters
    loss, grads = jax.value_and_grad(loss_fn)(weights_tuple, first_idxs, second_idxs, questions, labels)
    
    # Compute the Hessian
    hess = jax.hessian(loss_fn)(weights_tuple, first_idxs, second_idxs, questions, labels)
    
    # Extract the diagonal (hessian[i, j, i, j] for all i, j)
    hess_diag_embeddings = jnp.einsum('ijij->ij', hess[0][0])  # Diagonal of the embedding Hessian
    hess_diag_weight = jnp.einsum('ijij->ij', hess[1][1])  # Hessian for the weight matrix
    
    # Newton update: delta = grad / hessian_diag
    epsilon = 1e-6  # Small value to prevent division by zero
    delta_embeddings = grads[0] / (hess_diag_embeddings + epsilon)
    delta_weight = grads[1] / (hess_diag_weight + epsilon)
    
    # Update the embeddings and weight matrix
    new_embeddings = weights_tuple[0] - learning_rate * delta_embeddings
    new_weight = weights_tuple[1] - learning_rate * delta_weight
    
    return (new_embeddings, new_weight), loss
update_jitted = jax.jit(fun=update, static_argnames='learning_rate', donate_argnames='weights_tuple')

# Training loop
learning_rate = 1e-3
num_epochs = 10000
for epoch in range(num_epochs):
    # Update the weights tuple (embeddings and weight matrix) using the jitted update function
    weights_tuple, loss = update_jitted(weights_tuple, first_idxs, second_idxs, questions, labels, learning_rate)
    
    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Use second_order_embeddings for probability calculations
second_order_weights_tuple = weights_tuple
second_order_probabilities = compute_win_probabilities(second_order_weights_tuple)

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
    # ensures the folder exists
    output_folder.mkdir(exist_ok=True)
    # Convert the contestants_index dictionary to a list for ordering labels
    index_to_label = {v: k for k, v in contestants_index.items()}
    labels = [index_to_label[i] for i in range(len(index_to_label))]

    for judge, matrix in judges_dict.items():
        plt.figure(figsize=(8, 8))
        
        # Generate the heatmap with the transformed matrix for color scale but original values for annotation
        sns.heatmap(matrix, annot=matrix, fmt=".2f", cmap='coolwarm', xticklabels=labels, yticklabels=labels, cbar=False, vmin=0, vmax=1.0)
        
        # Add title and labels
        plt.title(f'Probabilities for Judge: {judge}')
        
        # Save the heatmap to the specified folder with the judge's name
        plt.savefig(output_folder / f'{judge}.png')
        
        # Close the plot to free up memory
        plt.close()

judges_dict = {f'human_1st_fit': first_order_probabilities, f'human_2nd_fit': second_order_probabilities}
output_folder = output_folder / 'fit'
plot_heatmaps(judges_dict, contestants_index, output_folder)