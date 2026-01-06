# reddit

## attribute_list = ["node_features", "node_degree", "clustering_coefficient"]

### with depth 3

balanced:
```
Nr.0 <= 1.0
├── yes?: Nr.0 <= 0.0
│   ├── yes?: it is a class 1! with no valid split is found with 6 v.s. 8
│   └── no?: it is a class 0! with no valid split is found with 153788 v.s. 137116
└── no?: Nr.0 <= 2.0
    ├── yes?: Nr.1 <= 0.0
    │   ├── yes?: it is a class 1! with 28583 v.s. 34241
    │   └── no?: it is a class 1! with 5925 v.s. 6013
    └── no?: Nr.1 <= 0.000403128273319453
        ├── yes?: it is a class 1! with 19664 v.s. 25714
        └── no?: it is a class 1! with 10036 v.s. 14910


```

unbalanced:

```
Nr.0 <= 1.0
├── yes?: Nr.0 <= 0.0
│   ├── yes?: it is a class 0! with no valid split is found with 13 v.s. 8
│   └── no?: it is a class 0! with no valid split is found with 452492 v.s. 137116
└── no?: Nr.0 <= 2.0
    ├── yes?: Nr.1 <= 0.0
    │   ├── yes?: it is a class 0! with 83797 v.s. 34241
    │   └── no?: it is a class 0! with 17657 v.s. 6013
    └── no?: Nr.1 <= 0.0004921259824186563
        ├── yes?: it is a class 0! with 57730 v.s. 25737
        └── no?: it is a class 0! with 29563 v.s. 14887

```
### depth 2:

balanced:

```
Nr.0 <= 1.0
├── yes?: Nr.0 <= 0.0
│   ├── yes?: it is a class 1! with 5 v.s. 8
│   └── no?: it is a class 0! with 153942 v.s. 137116
└── no?: Nr.0 <= 2.0
    ├── yes?: it is a class 1! with 34545 v.s. 40254
    └── no?: it is a class 1! with 29510 v.s. 40624

```

unbalanced:

```
Nr.0 <= 1.0
├── yes?: Nr.0 <= 0.0
│   ├── yes?: it is a class 0! with 13 v.s. 8
│   └── no?: it is a class 0! with 452492 v.s. 137116
└── no?: Nr.0 <= 2.0
    ├── yes?: it is a class 0! with 101454 v.s. 40254
    └── no?: it is a class 0! with 87293 v.s. 40624

```

# MUTAG
balanced
### depth 3:

```
Nr.0 <= 0.0
├── yes?: Nr.5 <= 0.0
│   ├── yes?: Nr.3 <= 0.0
│   │   ├── yes?: it is a class 0! with 289 v.s. 214
│   │   └── no?: it is a class 0! with 7 v.s. 1
│   └── no?: it is a class 0! with no valid split is found with 19 v.s. 1
└── no?: Nr.7 <= 1.0
    ├── yes?: it is a class 0! with no valid split is found with 21 v.s. 2
    └── no?: Nr.7 <= 2.0
        ├── yes?: it is a class 1! with 316 v.s. 339
        └── no?: it is a class 1! with 226 v.s. 321

```

### depth 2

```
Nr.0 <= 0.0
├── yes?: Nr.5 <= 0.0
│   ├── yes?: it is a class 0! with 296 v.s. 223
│   └── no?: it is a class 0! with 19 v.s. 2
└── no?: Nr.7 <= 1.0
    ├── yes?: it is a class 0! with 21 v.s. 1
    └── no?: it is a class 1! with 542 v.s. 652

```