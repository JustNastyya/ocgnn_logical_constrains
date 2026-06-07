import json

import pandas as pd
import os

from experiments.graph_level.data_loader_gl import TUDATASETS
from experiments.node_level.data_loader_nl import DATASET_REFERENCE


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
    input_path: str,
    output_filepath: str,
) -> pd.DataFrame:
    all_rows = []

    all_files = os.listdir(input_path + "results")
    
    for json_file in all_files:
        if json_file == ".ipynb_checkpoints":
            continue
        json_path = input_path + "results/" + json_file
        with open(json_path) as f:
            records = json.load(f)
        
        is_graph = any([dataset in json_file for dataset in TUDATASETS])
        is_node = any([dataset in json_file for dataset in DATASET_REFERENCE.keys()])
        
        if not(is_graph) and not(is_node):
            print(f"Unknown dataset {json_file}")
            continue

        for record in records:
            try:
                # finding the level
                if is_graph:
                    all_rows.append(_flatten_record(record, "graph"))
                
                elif is_node:
                    all_rows.append(_flatten_record(record, "node"))
            except Exception as e:
                print(f"Couldnt add an etry from json {json_file}")
                print(e)
                print(record)
            
    df = pd.DataFrame(all_rows)
    df.to_csv(output_filepath, index=False)
    print(f"Saved {len(df)} rows to {output_filepath}")
    return df


if __name__ == "__main__":
    experiment_name = "constraints_from_trees"

    # --------------- end
    input_path = f"wrapping_data/{experiment_name}/"
    output_filepath = f"{input_path}{experiment_name}_final.csv"

    generate_csv(input_path, output_filepath)
