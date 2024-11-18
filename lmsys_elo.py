# extracted from the official LMSYS google collab: 
# https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=3v8wc_oCmtmW
from pathlib import Path
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import json

# Download and preprocess data
# https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json
input_folder = Path("./data")
battles_file = input_folder / 'clean_battle_20240814_public.json'
battles = pd.read_json(battles_file).sort_values(ascending=True, by=["tstamp"])

# Filter for anonymous battles and deduplicated prompts
battles = battles[battles["anony"] == True]
battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]

# Function to compute Bradley-Terry Elo ratings
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = pd.pivot_table(
        df[df["winner"].isin(["tie", "tie (bothbad)"])],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = ptbl_tie + ptbl_tie.T
    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

# Compute Bradley-Terry Elo ratings
elo_ratings = compute_mle_elo(battles)

# Format results for JSON
elo_json = [{"Model": model, "Elo rating": rating} for model, rating in elo_ratings.items()]

# Save to a JSON file
with open(input_folder / "elo_ratings.json", "w") as outfile:
    json.dump(elo_json, outfile, indent=4)

print("Elo ratings saved to 'elo_ratings.json'")
