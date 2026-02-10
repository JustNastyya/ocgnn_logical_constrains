"""
This script takes the results of a bunch of training and says which
model has done better by comparing the test rate
"""
from itertools import product

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# =========================
# CONFIG
# =========================

# gl_compare_simple_OCGIN_vs_loss_constrains
JSON_NAME = "gl_compare_simple_OCGIN_vs_loss_specific_constrains" # without.json

JSON_PATH = f"experiments/graph_level/bunch_json_results/{JSON_NAME}.json"

VARYING_PARAMS_ALL = {"hidden_dim", "num_layers"}
VARYING_PARAMS_CONSTRAINS = {"constrains_filepath", "l_factor"}


def flatten_record(record):
    cfg = record["model_config"]
    res = record["model_results"]
    model_params = cfg["model_train"]
    
    flat = dict(cfg)
    flat.update(res)
    del flat["model_train"]
    flat.update(model_params)

    return flat

def get_dfs():
    # LOAD & FLATTEN JSON
    with open(JSON_PATH, "r") as f:
        raw = json.load(f)
    # record = raw[0]
    rows = [flatten_record(r) for r in raw]
    df = pd.DataFrame(rows)
    
    df = df[list(VARYING_PARAMS_ALL | VARYING_PARAMS_CONSTRAINS | {"is_logical", "test_rate"})]

    return df


df = get_dfs()

hidden_dim_unique = df.hidden_dim.unique().tolist()
num_layers_unique = df.num_layers.unique().tolist()

diff_test_rates = []
for hidden_dim, num_layers in product(
        hidden_dim_unique, num_layers_unique
    ):
    this_params_df = df.loc[(df.hidden_dim == hidden_dim) & (df.num_layers == num_layers)]
    
    normal_test_rate = this_params_df[this_params_df.is_logical == False].test_rate
    logical_test_rates = this_params_df[this_params_df.is_logical].test_rate
    
    diff = logical_test_rates - normal_test_rate.item()
    diff_test_rates.extend(diff.tolist())

print(sum(diff_test_rates) / len(diff_test_rates))