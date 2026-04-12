"""
This script takes the results of a bunch of training and says which
model has done better by comparing the test rate
"""
from itertools import product

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import plotly.express as px

# =========================
# CONFIG
# =========================

# gl_compare_simple_OCGIN_vs_loss_constrains
JSON_NAME = "test_nl_all" # without.json

JSON_PATH = f"experiments/node_level/bunch_json_results/{JSON_NAME}.json"

VARYING_PARAMS_ALL = {"num_layers"}
VARYING_PARAMS_CONSTRAINS = {"constrains_filepath", "l_factor", "train_loop"}

TARGET = "ROC_AUC"


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
    
    print(df.columns)
    df = df[list(VARYING_PARAMS_ALL | VARYING_PARAMS_CONSTRAINS | {"is_logical", TARGET})]

    return df


df = get_dfs()

num_layers_unique = df.num_layers.unique().tolist()

diff_test_rates = []
for num_layers in num_layers_unique:
    this_params_df = df.loc[df.num_layers == num_layers]
    
    normal_test_rate = this_params_df[this_params_df.is_logical == False][TARGET]
    logical_test_rates = this_params_df[this_params_df.is_logical][TARGET]
    diff = logical_test_rates - normal_test_rate.min().item() # TODO auf jedem jedem fall entfernen
    diff_test_rates.extend(diff.tolist())

print(df)
print(sum(diff_test_rates) / len(diff_test_rates))



def scatter_plot(df, level, output_dir):
    datasets = df["dataset"].unique()
    for dataset in datasets:
        this_data = df[df["dataset"] == dataset]
        logical_df = this_data[this_data["is_logical"]].copy()
        baseline_df = this_data[this_data["is_logical"] == False]

        for metric in METRICS:
            logical_df["model"] = logical_df["train_loop"].map(MODEL_LABELS)

            fig = px.scatter(
                logical_df,
                x="l_factor",
                y=metric,
                color="model",
                symbol="handler_short",
                facet_col="num_layers",
                log_x=True,
                title=f"Effect of λ on {metric} - {dataset}",
            )

            layers = sorted(this_data["num_layers"].unique())

            for i, layer in enumerate(layers):
                baseline_val = baseline_df.loc[
                    baseline_df["num_layers"] == layer, metric
                ].mean()

                fig.add_hline(
                    y=baseline_val,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="Baseline",
                    row=1,
                    col=i + 1
                )

            fig.update_traces(marker=dict(size=10))

            fig.for_each_xaxis(lambda axis: axis.update(title="λ"))

            fig.update_layout(
                xaxis_title="λ",
                yaxis_title=metric,
                legend_title="Model (Handler)"
            )

            fig.write_image(output_dir / f"{level}_{metric.lower()}_{dataset}_l_factor.png", width=1200, height=600)
            plt.close()

    print(f"Saved scatter plots for {level}")




import plotly.express as px

logical_df = df[df["is_logical"]]
baseline_df = df[df["is_logical"] == False]

# rename models for the legend
model_names = {
    "train_node_ocgin_add_loss_constrains": "Add Loss",
    "train_node_ocgin_weighting": "Weight Loss",
    "train_node_ocgin_irnoring_sus": "Ignore SUS",
}

logical_df = logical_df.copy()
logical_df["model"] = logical_df["train_loop"].map(model_names)

fig = px.scatter(
    logical_df,
    x="l_factor",
    y="ROC_AUC",
    color="model",
    facet_col="num_layers",
    log_x=True,
    title="Effect of λ on ROC-AUC",
)

# add baseline lines per layer
layers = sorted(df["num_layers"].unique())

for i, layer in enumerate(layers):
    baseline_auc = baseline_df.loc[
        baseline_df["num_layers"] == layer, "ROC_AUC"
    ].mean()

    fig.add_hline(
        y=baseline_auc,
        line_dash="dash",
        line_color="black",
        annotation_text="Baseline",
        row=1,
        col=i + 1
    )

fig.update_traces(marker=dict(size=10))

fig.for_each_xaxis(lambda axis: axis.update(title="λ"))

fig.update_layout(
    xaxis_title="λ",
    yaxis_title="ROC-AUC",
    legend_title="Model"
)

fig.write_image("wrapping_data/plots/test_nl_all.pdf", width=1000, height=500)