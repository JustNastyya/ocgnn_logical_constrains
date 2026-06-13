import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "constraints_from_trees" / "constraints_from_trees_final.csv"

POSITION_TO_MODEL = {
    0: ("logic_add", "loss"),
    1: ("logic_add", "loss"),
    2: ("logic_weight", "loss"),
    3: ("logic_weight", "loss"),
    4: ("logic_ignore_sus", "loss"),
    5: ("logic_ignore_sus", "loss"),
    6: ("logic_forecast_add", "inference"),
    7: ("logic_forecast_add", "inference"),
    8: ("logic_forecast_weight", "inference"),
    9: ("logic_forecast_weight", "inference"),
}


def recover_model_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_variant"] = None
    df["constrains_type"] = None

    baseline_mask = df["is_logical"] == False
    df.loc[baseline_mask, "model_variant"] = "baseline"

    logical = df[df["is_logical"] == True].copy()
    issues = []

    for (dataset, h_dim, n_layers, l_factor, dt_depth), group in logical.groupby(
        ["dataset", "hidden_dim", "num_layers", "l_factor", "dt_max_depth"],
        sort=False,
    ):
        group = group.sort_index()
        n = len(group)

        for pos, idx in enumerate(group.index):
            if pos > 9:
                issues.append(
                    f"Row idx={idx}: group ({dataset}, h={h_dim}, l={n_layers}, "
                    f"lf={l_factor}, dt={dt_depth}) has {n} rows, pos={pos} exceeds 9"
                )
                continue

            variant, ctype = POSITION_TO_MODEL.get(pos, (None, None))
            if variant:
                df.at[idx, "model_variant"] = variant
                df.at[idx, "constrains_type"] = ctype

    if issues:
        print(f"⚠ {len(issues)} issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ All constraints experiments fit into 10-row groups")

    n_assigned = df["model_variant"].notna().sum()
    print(f"\nAssigned model_variant to {n_assigned}/{len(df)} rows")

    for variant in sorted(df["model_variant"].dropna().unique()):
        count = (df["model_variant"] == variant).sum()
        print(f"  {variant}: {count}")

    return df


if __name__ == "__main__":
    print(f"Reading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    df = recover_model_types(df)

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved to {CSV_PATH}")
