# experiments aim

i want to create an automatic pipeline for experimenting. i want to be able to specify:

- dataset (for now i am by TUDataset)
- number of layers
- learning rate
- hidden layers
- and the model
- ggf the set of logical constrains
- epochs
- evaluation metrics

and start the experimant and save the logs

all models will be imorted saved in the model_reference where:

```
{
    "model_name": {
        "model": model_object,
        "train_loop": train_funciton
    },
    ...
}
```

for results are to be: 

Loss, test error

## after addinga new model:

- add model to experiments/model_reference
- change config in run_experiment

## after adding a new dataset:

- add dataset in experiments/data_loader

TODO:
- implement different evaluation metrics
- logical constrains, of course
- config as class in order not to fuck with strings
- from torch_geometric.loader import NeighborLoader for big datasets for node level
- gl: not just TUDatasets