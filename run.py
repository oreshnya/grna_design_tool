# In[]:

import numpy as np
import pandas as pd


# In[]:
### PAM search

from modules.core import (load_sequence, find_candidate_sgRNAs, filter_sgRNAs,
                          generate_features_in_r, predict_from_r_model, build_dataframe)

sequence = load_sequence("data/DAK1.fasta")
candidates = find_candidate_sgRNAs(sequence, pam="NGG")
print(f"Найдено {len(candidates)} sgRNA-кандидатов.")
print(candidates[:3])

filtered = filter_sgRNAs(candidates)
print(f"Осталось {len(filtered)} sgRNA после фильтрации.")

windows = [x["window"] for x in filtered]
features_df = generate_features_in_r(windows)

scores = predict_from_r_model(features_df)
df_sg_candidates = build_dataframe(filtered, features_df, scores)

print(df_sg_candidates.head())

# In[]:
# ### delta G

from modules.energy_calc import add_otSeq_column, compute_hybridization_energy

df_sg_candidates = add_otSeq_column(df_sg_candidates, sgrna_col="sgRNA_seq")
df_sg_candidates = compute_hybridization_energy(
    df_sg_candidates,
    sgrna_col="sgRNA_seq",
    otseq_col="otSeq"
)
print(df_sg_candidates[['sgRNA_seq', 'otSeq', 'deltaG_h']].head())


# In[]:
# ### CRIPSR-BERT

from models.CRISPR_BERT.model import build_bert

from modules.core import (load_off_targets_from_crispritz, prepare_crisprbert_df,
                          add_rnn_encoded_column, add_bert_encoding_columns, run_crisprbert_prediction)

# === Настройки ===
OFF_TARGETS_FROM_CRISPRITZ_PATH = "data/emx1.hg38.targets.txt"

# === Пайплайн ===
df = load_off_targets_from_crispritz(OFF_TARGETS_FROM_CRISPRITZ_PATH, n_rows=1000)
df = prepare_crisprbert_df(df)
df = add_rnn_encoded_column(df)
df = add_bert_encoding_columns(df)

WEIGHTS_PATH = "models/CRISPR_BERT/weight/I1.h5"

bert_model = build_bert()
bert_model.load_weights(WEIGHTS_PATH)

df_with_preds = run_crisprbert_prediction(df, bert_model, n_predict=-1)


# In[]:
# ### Relative activity predictor

import os
from models.RA_predictor.RA_predictor import build_relative_activity_predictor
from keras import backend as K

WEIGHTS_PATH_RA = os.path.join("models", "RA_predictor", "RA_predictor_1.weights.h5")

ra_model = build_relative_activity_predictor()
ra_model.load_weights(WEIGHTS_PATH_RA)