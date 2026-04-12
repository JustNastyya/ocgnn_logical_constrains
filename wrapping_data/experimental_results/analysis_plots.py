import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy.stats import mannwhitneyu

INPUT_CSV = Path(__file__).parent / "pure_loss_based" / "all_loss_all_experiments.csv"
OUTPUT_DIR = Path(__file__).parent / "pure_loss_based" / "plots"

METRICS = ["ROC_AUC"]

MODEL_LABELS = {
    "train_node_ocgin": "Baseline",
    "train_graph_ocgin": "Baseline",
    "train_node_ocgin_add_loss_constrains": "Add Loss",
    "train_graph_ocgin_add_loss_constrains": "Add Loss",
    "train_node_ocgin_weighting": "Weighting",
    "train_graph_ocgin_weighting": "Weighting",
    "train_node_ocgin_irnoring_sus": "Ignore SUS",
    "train_graph_ocgin_irnoring_sus": "Ignore SUS",
}


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def _short_handler(handler):
    if handler is None or pd.isna(handler):
        return None
    elif "Score" in str(handler):
        return "Distance based"
    else:
        return "Fuzzy logic"


def load_and_prepare_data():
    df = pd.read_csv(INPUT_CSV)
    df["model_type"] = df["train_loop"].map(MODEL_LABELS)
    df["is_logical_str"] = df["is_logical"].map({True: "Logical", False: "Non-Logical"})
    df["handler_short"] = df["constrains_handler"].apply(_short_handler)
    return df


def plot_comparison_boxplots(df, level, output_dir):
    level_df = df[df["experiment_level"] == level]
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=level_df, x="dataset", y=metric, hue="model_type", ax=ax)
        ax.set_title(f"{metric} by Dataset and Model Type ({level.capitalize()}-Level)")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Model Type")
        plt.tight_layout()
        fig.savefig(output_dir / f"{level}_{metric.lower()}_boxplot.png", dpi=150)
        plt.close()
    print(f"Saved comparison boxplots for {level}")

def plot_lfactor_effect(df, level, output_dir):
    logical_df = df[(df["experiment_level"] == level) & (df["is_logical"] == True)]
    if logical_df.empty:
        print(f"No logical experiments for {level}")
        return
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(
            data=logical_df,
            x="l_factor",
            y=metric,
            hue="dataset",
            style="constrains_handler",
            markers=True,
            ax=ax
        )
        ax.set_title(f"Effect of l_factor on {metric} ({level.capitalize()}-Level)")
        ax.set_xlabel("l_factor")
        ax.set_ylabel(metric)
        ax.set_xscale("log")
        ax.legend(title="Dataset / Handler", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(output_dir / f"{level}_{metric.lower()}_lfactor_effect.png", dpi=150)
        plt.close()
    print(f"Saved l_factor effect plots for {level}")


def scatter_plot(df, level, output_dir):
    level_df = df[df["experiment_level"] == level]
    datasets = level_df["dataset"].unique()
    for dataset in datasets:
        this_data = level_df[level_df["dataset"] == dataset]
        logical_df = this_data[this_data["is_logical"]].copy()
        baseline_df = this_data[this_data["is_logical"] == False]

        layers = sorted(this_data["num_layers"].unique())
        hidden_dims = sorted(this_data["hidden_dim"].unique())

        num_hidden_rows = len(hidden_dims)
        num_layer_cols = len(layers)

        metric = METRICS[0]
        
        logical_df["model"] = logical_df["train_loop"].map(MODEL_LABELS)

        fig = px.scatter(
            logical_df,
            x="l_factor",
            y=metric,
            color="model",
            symbol="handler_short",
            facet_row="hidden_dim",
            facet_col="num_layers",
            log_x=True,
            title=f"Effect of λ on {metric} - {dataset}",
        )

        for row_idx, hidden in enumerate(hidden_dims):
            for col_idx, layer in enumerate(layers):
                baseline_subset = baseline_df[
                    (baseline_df["num_layers"] == layer) &
                    (baseline_df["hidden_dim"] == hidden)
                ]
                if not baseline_subset.empty:
                    baseline_val = baseline_subset[metric].mean()
                    fig.add_hline(
                        y=baseline_val,
                        line_dash="dash",
                        line_color="black",
                        annotation_text="Baseline",
                        row=row_idx + 1,
                        col=col_idx + 1
                    )

        fig.update_traces(marker=dict(size=8))

        fig.for_each_xaxis(lambda axis: axis.update(title="λ"))

        fig.update_layout(
            xaxis_title="λ",
            yaxis_title=metric,
            legend_title="Model (Handler)"
        )

        width = 250 * num_layer_cols
        height = 200 * num_hidden_rows
        fig.write_image(output_dir / f"{level}_{metric.lower()}_{dataset}_l_factor.png", width=width, height=height)
        plt.close()

    print(f"Saved scatter plots for {level}")


def plot_architecture_heatmap(df, output_dir):
    agg_df = df.groupby(["model_type", "hidden_dim", "num_layers"])["ROC_AUC"].mean().reset_index()
    
    model_types = sorted(agg_df["model_type"].unique())
    num_models = len(model_types)
    
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 6), sharey=True)
    
    for i, model in enumerate(model_types):
        model_df = agg_df[agg_df["model_type"] == model]
        pivot = model_df.pivot(index="hidden_dim", columns="num_layers", values="ROC_AUC")
        
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="viridis",
            ax=axes[i], cbar=i == num_models - 1
        )
        axes[i].set_title(f"{model}")
        axes[i].set_xlabel("Num Layers")
        if i == 0:
            axes[i].set_ylabel("Hidden Dim")
    
    plt.suptitle("Mean ROC_AUC by Architecture (hidden_dim x num_layers)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "architecture_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved architecture heatmap")


def plot_handler_comparison_overall(df, output_dir):
    df_handlers = df[df["handler_short"].notna()]
    if df_handlers.empty:
        print("No handler data")
        return
    
    metric = METRICS[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_handlers, x="handler_short", y=metric, ax=ax)
    ax.set_title(f"Handler Comparison Overall")
    ax.set_xlabel("Handler")
    ax.set_ylabel(metric)
    plt.tight_layout()
    fig.savefig(output_dir / "handler_comparison_overall.png", dpi=150)
    plt.close()
    
    fuzzy = df_handlers[df_handlers["handler_short"] == "Fuzzy logic"][metric]
    distance = df_handlers[df_handlers["handler_short"] == "Distance based"][metric]
    stat, pval = mannwhitneyu(fuzzy.dropna(), distance.dropna(), alternative="two-sided")
    print(f"Handler comparison: Fuzzy median={fuzzy.median():.4f}, Distance median={distance.median():.4f}, p-value={pval:.4e}")
    print("Saved handler comparison overall")


def main():
    setup_style()

    node_dir = OUTPUT_DIR
    graph_dir = OUTPUT_DIR
    node_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()
    print(f"Loaded {len(df)} experiments")

    for level, out_dir in [("node", node_dir), ("graph", graph_dir)]:
        plot_comparison_boxplots(df, level, out_dir)
        plot_lfactor_effect(df, level, out_dir)
        # scatter_plot(df, level, out_dir)

    plot_architecture_heatmap(df, OUTPUT_DIR)
    plot_handler_comparison_overall(df, OUTPUT_DIR)


    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
