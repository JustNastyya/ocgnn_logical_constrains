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

for results are to be: Loss, test error

you can specialize if you want to automatically save logs or not.

## constrains:

after adding a new constrains file in `constrains/data` you need to specialize the constrain handler, constrains filepath and the l_factor

## after addinga new model:

- add model to experiments/model_reference
- change config in run_experiment

## after adding a new dataset:

- add dataset in experiments/data_loader

# running bunch of experiments

if you want to iterate through config parameters change stuff in the tun_bunch_experiments and run it. you can change:
- list of hidden dims
- list of l_factor 
- list of num_layers

the script will automatically train models with all of these parameters and save the results in the specifies .json file

TODO:
- implement different evaluation metrics
- config as class in order not to fuck with strings
- from torch_geometric.loader import NeighborLoader for big datasets for node level
- gl: not just TUDatasets
