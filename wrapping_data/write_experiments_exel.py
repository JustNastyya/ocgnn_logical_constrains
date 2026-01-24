"""
This script is written by chatgpt cuz i dont give a shit
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# =========================
# CONFIG
# =========================
JSON_PATH = "experiments/node_level/bunch_json_results/first_try.json"
OUTPUT_EXCEL = "wrapping_data/exel/model_results_first_try.xlsx"
METRIC_NAME = "test_rate"

VARYING_PARAMS = {"hidden_dim", "num_layers", "constrains_l"}

# =========================
# HELPERS
# =========================
def extract_constrains_l(constrains_filepath):
    """
    Extract constrains_l from filepath.
    Example:
    Cora_auto_generated_3_101_102_103.json -> 101_102_103
    """
    if not constrains_filepath:
        return None
    name = Path(constrains_filepath).stem
    parts = name.split("_")
    return "_".join(parts[4:]) if len(parts) > 4 else name


def flatten_record(record):
    cfg = record["model_config"]
    res = record["model_results"]

    flat = dict(cfg)
    flat.update(res)

    flat["constrains_l"] = extract_constrains_l(
        cfg.get("constrains_filepath")
    )

    return flat


# =========================
# LOAD & FLATTEN JSON
# =========================
with open(JSON_PATH, "r") as f:
    raw = json.load(f)

rows = [flatten_record(r) for r in raw]
df = pd.DataFrame(rows)

# =========================
# FIND NON-VARYING PARAMS
# =========================
non_varying = {}
for col in df.columns:
    if col in VARYING_PARAMS or col == METRIC_NAME:
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
        sheet_name = f"hidden_dim_{hidden_dim}"
        start_row = 0

        # =========================
        # NON-LOGICAL MODELS
        # =========================
        subset_non_logical = df_non_logical[
            df_non_logical["hidden_dim"] == hidden_dim
        ]

        if not subset_non_logical.empty:
            pivot_non_logical = subset_non_logical.pivot_table(
                index="num_layers",
                values=METRIC_NAME,
                aggfunc="mean"
            ).reset_index()

            pivot_non_logical.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=start_row
            )

            start_row += len(pivot_non_logical) + 3

        # =========================
        # LOGICAL MODELS
        # =========================
        subset_logical = df_logical[
            df_logical["hidden_dim"] == hidden_dim
        ]

        for constrains_l in sorted(
            subset_logical["constrains_l"].dropna().unique()
        ):
            subset_c = subset_logical[
                subset_logical["constrains_l"] == constrains_l
            ]

            if subset_c.empty:
                continue

            title_df = pd.DataFrame(
                [[f"Constrains: {constrains_l}"]],
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

            pivot_logical = subset_c.pivot_table(
                index="num_layers",
                values=METRIC_NAME,
                aggfunc="mean"
            ).reset_index()

            pivot_logical.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=start_row
            )

            start_row += len(pivot_logical) + 3

print(f"Excel file written to: {OUTPUT_EXCEL}")
