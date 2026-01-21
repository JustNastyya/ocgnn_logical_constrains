ATTRIBUTE_CODE = {
    "node_features": "101",
    "node_degree": "102",
    "clustering_coefficient": "103",
}

def get_filename(dataset_name, attribute_list, max_depth):
    coded_attributes = []
    for attribute in attribute_list:
        coded_attributes.append(ATTRIBUTE_CODE[attribute])

    coded_attributes_str = "_".join(coded_attributes)
    filepath = f"constrains/data/{dataset_name}_auto_generated_{max_depth}_{coded_attributes_str}.json"
    return filepath