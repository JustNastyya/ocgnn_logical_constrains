import json

import pandas as pd


def _flatten_record(record: dict, experiment_level: str) -> dict:
    flat = {"experiment_level": experiment_level}

    model = record.get("model_config", {})
    res = record.get("model_results", {})
    
    flat.update(model)
    flat.update(res)

    model_train = model.get("model_train", {})
    flat["model"] = model_train.get("model")
    flat["train_loop"] = model_train.get("train_loop")
    del flat["model_train"]
    
    if record["model_config"]["is_logical"]:
        tree = model.get("decision_tree", {})
        flat["dt_max_depth"] = tree.get("max_depth")
        flat["dt_attribute_list"] = ", ".join(tree.get("attribute_list", []))
        del flat["decision_tree"]

    return flat


def generate_csv(
    node_datasets: list[str],
    graph_datasets: list[str],
    input_path: str,
    input_file: str,
    output_filepath: str,
) -> pd.DataFrame:
    all_rows = []

    for dataset in node_datasets:
        json_path = input_path + f"results/{input_file}_nl_{dataset}.json"
        with open(json_path) as f:
            records = json.load(f)
        for record in records:
            try:
                all_rows.append(_flatten_record(record, "node"))
            except Exception as e:
                print(f"Couldnt add an etry from dataset {dataset}")
                print(e)
            
    for dataset in graph_datasets:
        json_path = input_path + f"results/{input_file}_gl_{dataset}.json"
        with open(json_path) as f:
            records = json.load(f)
        for record in records:
            try:
                all_rows.append(_flatten_record(record, "graph"))
            except Exception as e:
                print(f"Couldnt add an etry from dataset {dataset}")
                print(e)
            
    df = pd.DataFrame(all_rows)
    df.to_csv(output_filepath, index=False)
    print(f"Saved {len(df)} rows to {output_filepath}")
    return df


if __name__ == "__main__":
    
    # ------------- TO MODIFY if needed
    """
    node_datasets = [
        "CiteSeer", "Computers", "Cora", "Cornell", "CS",
        "Photo", "PubMed", "Texas", "Wisconsin",
    ]
    """
    """
    graph_datasets = [
        "AIDS", "COIL-RAG", "DD", "DHFR", "ENZYMES",
        "MSRC_21", "MUTAG", "PC-3H", "PROTEINS", "YeastH",
    ]
    """
    graph_datasets = []
    node_datasets = ["Cora", "CiteSeer", "Cornell", "PubMed"]
    
    input_path = "wrapping_data/experimental_results/constrains_forecasting/"
    input_file = "loss_forecasting"
    
    # --------------- end
    
    output_filepath = f"{input_path}{input_file}_all_experiments.csv"

    generate_csv(node_datasets, graph_datasets, input_path, input_file, output_filepath)
