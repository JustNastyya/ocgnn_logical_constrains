# this is a constrains handler

it is designed to implement constrains fast and robust.

we have handlers which load constrains and use them.

## data
the constrains are saved under `data` and have a structure of:

all conditions in the condition list are AND to each other to create the "predicted class"

additional_attributes describe the additional attributes and their used indexes in the constrains
```
{
    "constrains": [
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
    ],
    "additional_attributes": {
        "1433": "node_degree",
        "1434": "clustering_coefficient"
  }
}
```

## handlers

`NLRuleBasedHandler`

implements constrains on a given data (nodal attributes matrix)

implements soft functions as:

`a <= b` is `sigmoid(lambda * (b - a))`

`a > b` is `sigmoid(lambda * (a - b))`

`a AND b` is `a * b`

`a OR b` is `1 - (1 - a) * (1 - b)`

`get_constraint_value` returns the implementation of all constrains for a given x