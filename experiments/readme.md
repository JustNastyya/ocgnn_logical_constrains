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

TODO:
- implement different evaluation metrics
- logical constrains, of course
- why is R one dimentional 1 number? !!!!!!
- config as class in order not to fuck with strings