# Judge Evaluator

Statistic to decide on which LLM best emulates a human picking between chatbots.

We observed that the naive approach of seing how often a judging model agrees with a human is noisy (there is little human data compared to all permutations of models and questions) and that it, as well as more clever variant (such as looking at the ELO scores given to the model) puts equal emphasis on all comparison / models while we only care about the judge's ability to pick the best model.

Our key insight is the idea that, in practice, we pick a new model by comparing it to the current best model on a serie of questions.
What we want is the judge which, given the same process, is most likely to end up with *the same best final model* as the human.
The following is a statistical formalization of that intuition, examining the distribution of final models.

## Theory

Historical contest data tells us, given a judge (which might be a human), what is the probability of one model doing better than another on a given question.

Based on this, we compute the probability of one model beating another on a majority of the question in a trial (default to 20 questions).

We model starting with a model, comparing it to a random other model in a trial, and keeping the best one (10 times in a row, we allow trying agin against the same model).

We compute this using a discrete markov chain, giving us a distribution of end models: the probability that a given model would be considered the best after the trials.

Alignement between two judges (their distribution) could be done with a simple dot product but it suppose non correlation which is very pessimistic.
Instead we suppose maximal correlation and compute it as the sum of the minimum of the paired probability of the models.
This gives use the probability that the human and the judge would end up with the same end winner.

## Usage

### Install

Run the following to install the code and its dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install scipy matplotlib seaborn scikit-learn
python3 -m pip install -U "jax[cuda12]"
```

### Run

run the following to run the code on the data:

```sh
source venv/bin/activate
python3 evaluator.py
```
