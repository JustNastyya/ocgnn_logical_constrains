import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# =========================
# CONFIG
# =========================


JSON_NAME = "nl_presentation_node_specific_take_0" # without.json

JSON_PATH = f"experiments/node_level/bunch_json_results/{JSON_NAME}.json"
OUTPUT_EXCEL = f"wrapping_data/exel/model_results_{JSON_NAME}.xlsx"

METRIC_NAME = "balanced_accuracy"
VARYING_PARAMS_ALL = {"hidden_dim", "num_layers"}
VARYING_PARAMS_CONSTRAINS = {"constrains_filepath", "l_factor"}
SHEET_PARAM = "num_layers"

"""
for now works only for these varying parameters

num_layers is assigned to the sheets

on every sheet:
1 table: non constrains
hidden dim - the col of the table

1 table for every constrains_filepath with
col: hidden_dim, row - l_factor

"""


def flatten_record(record):
    cfg = record["model_config"]
    res = record["model_results"]
    model_params = cfg["model_train"]
    
    flat = dict(cfg)
    flat.update(res)
    del flat["model_train"]
    flat.update(model_params)

    return flat


# =========================
# LOAD & FLATTEN JSON
# =========================
with open(JSON_PATH, "r") as f:
    raw = json.load(f)
# record = raw[0]
rows = [flatten_record(r) for r in raw]
df = pd.DataFrame(rows)

# =========================
# FIND NON-VARYING PARAMS
# =========================
non_varying = {}
for col in df.columns:
    if col in VARYING_PARAMS_ALL or col in VARYING_PARAMS_CONSTRAINS or col == METRIC_NAME:
        continue

    # skip columns with dicts / lists
    try:
        unique_count = df[col].nunique(dropna=False)
    except TypeError:
        continue

    if unique_count == 1:
        non_varying[col] = df[col].iloc[0]



non_varying_df = pd.DataFrame(
    list(non_varying.items()),
    columns=["Parameter", "Value"]
)

# =========================
# SPLIT LOGICAL / NON-LOGICAL
# =========================
df_non_logical = df[df["is_logical"] == False]
df_logical = df[df["is_logical"] == True]

# =========================
# WRITE EXCEL
# =========================
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    # ---- Sheet 0: Non-varying parameters ----
    non_varying_df.to_excel(
        writer,
        sheet_name="Model_Info",
        index=False
    )

    # ---- One sheet per hidden_dim ----
    for hidden_dim in sorted(df["hidden_dim"].unique()):
        # hidden_dim = sorted(df["hidden_dim"].unique())[0]
        sheet_name = f"hidden_dim_{hidden_dim}"
        start_row = 0

        # =========================
        # NON-LOGICAL MODELS
        # =========================
        subset_non_logical = df_non_logical[
            df_non_logical["hidden_dim"] == hidden_dim
        ]

        pivot_non_logical = subset_non_logical[
            ["model", "num_layers", METRIC_NAME]
            ].set_index("model").T

        pivot_non_logical.to_excel(
            writer,
            sheet_name=sheet_name,
            index=True,
            startrow=start_row
        )

        start_row += len(pivot_non_logical) + 3

        # =========================
        # LOGICAL MODELS
        # =========================
        subset_logical = df_logical[
            df_logical["hidden_dim"] == hidden_dim
        ]

        # constrains_filepath = subset_logical["constrains_filepath"].loc[1]
        for constrains_filepath in sorted(
            subset_logical["constrains_filepath"].dropna().unique()
        ):
            subset_c = subset_logical[
                subset_logical["constrains_filepath"] == constrains_filepath
            ]

            if subset_c.empty:
                continue
            
            model_name = subset_c.model.reset_index(drop=True)[0]
            title_df = pd.DataFrame(
                [[f"Model: {model_name}, Constrains: {constrains_filepath}"]],
                columns=[""]
            )
            title_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=start_row,
                header=False
            )
            start_row += 1

            pivot_logical = subset_c.pivot(
                index="l_factor",        # rows
                columns="num_layers",    # columns
                values=METRIC_NAME       # cell values
            )

            pivot_logical.to_excel(
                writer,
                sheet_name=sheet_name,
                index=True,
                startrow=start_row
            )

            start_row += len(pivot_logical) + 3

print(f"Excel file written to: {OUTPUT_EXCEL}")
