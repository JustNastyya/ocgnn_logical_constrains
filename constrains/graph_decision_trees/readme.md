# Graph decision trees

Main idea is to implement graph decision trees in order to generate simple logical constrains to use them later in OCGIN.

For every level a specialized model adn config are spezified.

the config is a class implementing feature extraction for every node

there a list of node features is to be defined

the results are automatically saved in `constrains/dataset_tree_depth.json`

## result structure

all conditions in the condition list are AND to each other to create the "predicted class"
```
[
  {
    "conditions": [
      {
        "feature_index": ,
        "op": "<=",                     # or <
        "threshold": 0.0                # decision boundary
      },
      {
        "feature_index": 205,
        "op": "<=",
        "threshold": 0.0
      }
    ],
    "predicted_class": 1,
    "meta": {
      "depth": 2,
      "y_label_number": "98 v.s. 225"   # how hard is the descision boundary
    }
  },
...
]
```

so 1 tree of depth 3 can have up to 8 conditions

## node level

tu run execute `train_and_print.py`.

You can specialize:

- dataset
- features in `attribute_list`
- max depth

available features:
- `node_degree`
- `clustering_coefficient`

some of the results are saved in `tud_results.md` 

