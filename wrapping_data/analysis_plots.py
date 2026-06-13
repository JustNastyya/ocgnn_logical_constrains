import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu

experiment_name = "constraints_from_trees"
csv_name = "constraints_from_trees_final.csv"

INPUT_CSV = Path(__file__).parent / experiment_name / csv_name
OUTPUT_DIR = Path(__file__).parent / experiment_name / "plots"

METRICS = ["ROC_AUC"]

MODEL_LABELS = {
    "baseline": "Baseline",
    "logic_add": "Loss-Add",
    "logic_weight": "Loss-Weighting",
    "logic_ignore_sus": "Loss-Suppression",
    "logic_forecast_add": "Inference-Add",
    "logic_forecast_weight": "Inference-Weighting",
}

MODEL_ORDER = ["Baseline", "Loss-Add", "Loss-Weighting", "Loss-Suppression", "Inference-Add", "Inference-Weighting"]

PALETTE = {
    "Baseline": "#2c3e50",
    "Loss-Add": "#27ae60",
    "Loss-Weighting": "#2980b9",
    "Loss-Suppression": "#8e44ad",
    "Inference-Add": "#e74c3c",
    "Inference-Weighting": "#f39c12",
}


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12


def _short_handler(handler):
    if handler is None or pd.isna(handler):
        return None
    elif "Score" in str(handler) or "Distance" in str(handler):
        return "Distance"
    else:
        return "Fuzzy"


def load_and_prepare_data():
    df = pd.read_csv(INPUT_CSV)
    df["model_type"] = df["model_variant"].map(MODEL_LABELS)
    df["handler_short"] = df["constrains_handler"].apply(_short_handler)
    # mark loss-only vs inference
    df["inference_group"] = df["constrains_type"].map({
        "loss": "Loss-Only",
        "inference": "Inference-Time"
    })
    return df


def plot_comparison_boxplots(df, level, output_dir):
    level_df = df[df["experiment_level"] == level]
    metric = METRICS[0]

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.boxplot(
        data=level_df,
        x="dataset", y=metric,
        hue="model_type", hue_order=MODEL_ORDER,
        palette=PALETTE, ax=ax,
    )
    ax.set_title(f"ROC-AUC by Dataset and Model Variant ({level.capitalize()}-Level)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / f"{level}_roc_auc_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison boxplots for {level}")


def plot_lfactor_effect(df, level, output_dir):
    logical_df = df[(df["experiment_level"] == level) & (df["is_logical"] == True)]
    baseline_val = df[(df["experiment_level"] == level) & (df["model_variant"] == "baseline")]["ROC_AUC"].mean()

    metric = METRICS[0]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=logical_df,
        x="l_factor", y=metric,
        hue="model_type", hue_order=MODEL_ORDER[1:],
        style="model_type", style_order=MODEL_ORDER[1:],
        palette=PALETTE,
        markers=True, dashes=False, ax=ax,
    )
    ax.axhline(y=baseline_val, color="black", linestyle="--", linewidth=1.5, label="Baseline")
    ax.set_xscale("log")
    ax.set_title(f"Effect of λ on ROC-AUC ({level.capitalize()}-Level)")
    ax.set_xlabel("λ (constraint scaling factor)")
    ax.set_ylabel("Mean ROC-AUC")
    ax.legend(title="Model Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / f"{level}_lambda_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved lambda effect plot for {level}")


def plot_lambda_by_inference_type(df, level, output_dir):
    """Loss-only vs Inference-Time λ comparison."""
    logical_df = df[(df["experiment_level"] == level) & (df["is_logical"] == True)]
    metric = METRICS[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=logical_df,
        x="l_factor", y=metric,
        hue="inference_group",
        style="inference_group",
        markers=True, dashes=False, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title(f"Loss-Only vs Inference-Time: λ Effect ({level.capitalize()}-Level)")
    ax.set_xlabel("λ (constraint scaling factor)")
    ax.set_ylabel("Mean ROC-AUC")
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(output_dir / f"{level}_lambda_loss_vs_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved λ loss vs inference plot for {level}")


def plot_loss_vs_inference_comparison(df, level, output_dir):
    """Boxplot: Baseline vs all Loss-Only vs all Inference per dataset."""
    level_df = df[df["experiment_level"] == level]
    metric = METRICS[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=level_df,
        x="dataset", y=metric,
        hue="inference_group",
        ax=ax,
    )
    ax.set_title(f"Loss-Only vs Inference-Time vs Baseline ({level.capitalize()}-Level)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(output_dir / f"{level}_loss_vs_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss vs inference comparison for {level}")


def plot_handler_comparison(df, output_dir):
    handlers = df[df["handler_short"].notna()].copy()
    metric = METRICS[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=handlers,
        x="model_type", y=metric,
        hue="handler_short",
        hue_order=["Fuzzy", "Distance"],
        order=MODEL_ORDER[1:],
        ax=ax,
    )
    ax.set_title("Fuzzy vs Distance Handler by Model Variant")
    ax.set_xlabel("Model Variant")
    ax.set_ylabel("ROC-AUC")
    ax.legend(title="Handler")
    plt.tight_layout()
    fig.savefig(output_dir / "handler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    for model in MODEL_ORDER[1:]:
        fuzz = handlers[(handlers["model_type"] == model) & (handlers["handler_short"] == "Fuzzy")][metric].dropna()
        dist = handlers[(handlers["model_type"] == model) & (handlers["handler_short"] == "Distance")][metric].dropna()
        if len(fuzz) > 0 and len(dist) > 0:
            stat, pval = mannwhitneyu(fuzz, dist, alternative="two-sided")
            print(f"  {model}: Fuzzy median={fuzz.median():.4f}, Distance median={dist.median():.4f}, p-value={pval:.4e}")
    print("Saved handler comparison")


def plot_architecture_heatmap(df, output_dir):
    agg_df = df.groupby(["model_type", "hidden_dim", "num_layers"])["ROC_AUC"].mean().reset_index()

    models = [m for m in MODEL_ORDER if m in agg_df["model_type"].unique()]
    num_models = len(models)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes_flat = axes.flatten() if num_models > 1 else [axes]

    for i, model in enumerate(models):
        model_df = agg_df[agg_df["model_type"] == model]
        pivot = model_df.pivot(index="hidden_dim", columns="num_layers", values="ROC_AUC")
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="viridis",
            ax=axes_flat[i], cbar=i == 0,
        )
        axes_flat[i].set_title(f"{model}")
        axes_flat[i].set_xlabel("Layers")
        if i % cols == 0:
            axes_flat[i].set_ylabel("Hidden Dim")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("Mean ROC-AUC by Architecture (hidden_dim × layers)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "architecture_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved architecture heatmap")


def plot_best_config_highlights(df, output_dir):
    """Bar chart: baseline mean vs best constraint config per dataset."""
    metric = METRICS[0]
    rows = []
    for ds in sorted(df["dataset"].unique()):
        dsd = df[df["dataset"] == ds]
        baseline_mean = dsd[dsd["model_variant"] == "baseline"][metric].mean()
        best_row = dsd.loc[dsd[metric].idxmax()]
        rows.append({
            "dataset": ds,
            "baseline_mean": baseline_mean,
            "best_value": best_row[metric],
            "best_model": best_row["model_type"],
            "best_hd": best_row["hidden_dim"],
            "best_layers": best_row["num_layers"],
            "best_lf": best_row["l_factor"],
            "best_handler": best_row["handler_short"],
            "improvement": best_row[metric] - baseline_mean,
        })
    result_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(result_df))
    w = 0.35
    bars1 = ax.bar(x - w / 2, result_df["baseline_mean"], w, label="Baseline Mean", color="#7f8c8d")
    bars2 = ax.bar(x + w / 2, result_df["best_value"], w, label="Best Constraint Config", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(result_df["dataset"], rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Baseline Mean vs Best Constraint Configuration per Dataset")
    ax.legend()

    for i, row in result_df.iterrows():
        imp = row["improvement"]
        if row["best_value"] > row["baseline_mean"]:
            ax.annotate(f"+{imp:.3f}", (x[i], row["best_value"]),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color="green")
    plt.tight_layout()
    fig.savefig(output_dir / "best_config_highlights.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Best configs per dataset:")
    for _, row in result_df.iterrows():
        print(f"  {row['dataset']:15s} baseline={row['baseline_mean']:.4f} best={row['best_value']:.4f} ({row['best_model']:20s} hd={row['best_hd']} layers={row['best_layers']} λ={row['best_lf']} {row['best_handler']})")
    print("Saved best config highlights")


def plot_model_ranking_table(df, output_dir):
    """Table-like heatmap: mean ROC-AUC per dataset × model variant."""
    metric = METRICS[0]
    pivot = df.pivot_table(
        index="dataset", columns="model_type",
        values=metric, aggfunc="mean"
    )
    pivot = pivot[MODEL_ORDER]
    pivot = pivot.round(4)

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0.6,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Mean ROC-AUC"})
    ax.set_title("Mean ROC-AUC per Dataset and Model Variant")
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Model Variant")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_dir / "model_ranking_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model ranking heatmap")


def save_summary_statistics(df, output_dir):
    """Print and save summary statistics as a text file."""
    metric = METRICS[0]
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT SUMMARY STATISTICS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total experiments: {len(df)}")
    lines.append("")

    lines.append("--- Mean ROC-AUC by Model Variant ---")
    for m in MODEL_ORDER:
        subset = df[df["model_type"] == m]
        mean_val = subset[metric].mean()
        std_val = subset[metric].std()
        lines.append(f"  {m:25s} mean={mean_val:.4f}  std={std_val:.4f}")
    lines.append("")

    lines.append("--- Mean ROC-AUC by Inference Group ---")
    lines.append(f"  Loss-Only (training only):    {df[df['constrains_type']=='loss'][metric].mean():.4f}")
    lines.append(f"  Inference-Time:               {df[df['constrains_type']=='inference'][metric].mean():.4f}")
    lines.append(f"  Baseline:                     {df[df['model_variant']=='baseline'][metric].mean():.4f}")
    lines.append("")

    lines.append("--- Handler Comparison ---")
    handlers = df[df["handler_short"].notna()]
    fuzz = handlers[handlers["handler_short"] == "Fuzzy"][metric]
    dist = handlers[handlers["handler_short"] == "Distance"][metric]
    stat, pval = mannwhitneyu(fuzz.dropna(), dist.dropna(), alternative="two-sided")
    lines.append(f"  Fuzzy median={fuzz.median():.4f}, Distance median={dist.median():.4f}, p-value={pval:.4e}")
    lines.append("")

    lines.append("--- Pairwise Mann-Whitney U Tests ---")
    for i, m1 in enumerate(MODEL_ORDER):
        for m2 in MODEL_ORDER[i+1:]:
            s1 = df[df["model_type"] == m1][metric].dropna()
            s2 = df[df["model_type"] == m2][metric].dropna()
            stat, pval = mannwhitneyu(s1, s2, alternative="two-sided")
            sig = "SIGNIFICANT" if pval < 0.05 else "n.s."
            lines.append(f"  {m1:25s} vs {m2:25s}  U={stat:.0f}  p={pval:.6f}  [{sig}]")

    out_path = output_dir / "summary_statistics.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary statistics to {out_path}")
    for line in lines:
        print(line)





def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_data()
    print(f"Loaded {len(df)} experiments")

    for level in ("node", "graph"):
        print(f"\n{'=' * 60}")
        print(f"PLOTTING FOR {level.upper()}-LEVEL")
        print(f"{'=' * 60}")
        plot_comparison_boxplots(df, level, OUTPUT_DIR)
        plot_lfactor_effect(df, level, OUTPUT_DIR)
        plot_lambda_by_inference_type(df, level, OUTPUT_DIR)
        plot_loss_vs_inference_comparison(df, level, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("PLOTTING AGGREGATED FIGURES")
    print(f"{'=' * 60}")
    plot_architecture_heatmap(df, OUTPUT_DIR)
    plot_handler_comparison(df, OUTPUT_DIR)
    plot_best_config_highlights(df, OUTPUT_DIR)
    plot_model_ranking_table(df, OUTPUT_DIR)
    save_summary_statistics(df, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
