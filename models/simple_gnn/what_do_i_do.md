## general aim

so i want to build a ocgnn model with logical constrains.
this time let it be graph level or whatever.

and i want to compare it to a model without constrains.

this means i need to build a usual model first and then extend it with my constrains.

## my plan

1. find data
2. prepare it for training
3. train an usual model for this data and evaluate it
4. make up constrains with binary tree
5. build in constrains
6. train the model with constrains
7. compare models

## sound alright, yeah?

## my panic?..

1. how the fuck does my data look like
2. what bib should i use?
3. how can i look at my data?
4. what model exactly do i need?
5. how exactly can i train it?
6. how the fuck can i build a desicion tree on graphs
7. how the fuck do i implement logical constrains


# alright lets deal with all of it

## dataset

In the REDDIT datasets, each graph represents
a discussion thread, where nodes correspond to users, two
of which are connected by an edge if one responded to a
comment of the other. This graph model is used to derive
several datasets, where the classification task is to distinguish either discussion-based and question/answer-based subreddits (REDDIT-BINARY) or predict the subreddit, where
the thread was posted (REDDIT-MULTI-5K and REDDIT-
MULTI-12K)

stuff about data:

 - so a batch has a bunch of graphs. and it has numerated all graph nodes together 

1. after batch.edge_index:

```tensor([[   0,    0,    0,  ..., 7869, 7870, 7870], batch.edge_index
        [  28,   42,   78,  ..., 7671, 7629, 7726]])
```
means node 0 is connected to node 28 etc

2. after batch.batch
```tensor([ 0,  0,  0,  ..., 31, 31, 31])```
means 0-28 edge is about the graph nr. 0 etc

3. after batch.y
```
tensor([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 0])
```
is the label per graph

```
batch.x             - all node features from all graphs. if none - no node features
batch.edge_index    - all edges from all graphs
batch.batch         - a vector mapping each node to its graph ID
batch.y             - the labels for each graph (size = batch_size)
```

## what model

so i have OCGIN and OCGTL. both are for anomaly detection

- OCGIN - one class gnn based approach. it maps “normal” training graphs into an embedding space and treats anomalies as outliers (outside a learnt “normal” region)
- OCGTL - more advanced. A more advanced method that combines the one-class (deep OCC) objective with a self-supervised “transformation learning” objective. That is: besides embedding graphs, it performs additional learning with multiple GNNs / “views” to encourage meaningful diversity in the embeddings.

OCGTL avoids hypershere collapse, has better detection perfomance and is more robust.

but since i am a noob, lets start with 

## OCGIN (One-Class Graph Isomorphism Network)

- train on normal graphs
- use GIN graph encoder
- graph level embedding. To classify an entire graph, you need a fixed-size representation.
- One-Class Objective: Deep SVDD OCGIN uses the Deep SVDD (Support Vector Data Description) idea

embedding (for dummies) - is just a mapping form.

fuck it, lets train a simple gnn for now.

# hey cool combi!
by pressing ctrl + ` you can go to terminal.
by ctrl + 1 you can go to the opened file
by ctrl + 2 you can go to the opened file on the right (or open new one)

## pytorch basics

PyTorch is mainly about tensors and neural networks

a tensor is like  a multidimentional array, but it can live on a GPU

### neural networks
in pytoch all neural networks are subclasses of nn.Module
this class keeps track of the layers, applies forward functions usw.
when subclassing i need to define 2 things:

- layers in __init__
- forward computation in forward
(functions)

1. in init:

do the layer as `self.something = nn.layer_type(params like dimentions)`

2. in forward:

defy how the data flows. like you get in `forward(self, x)`
and x is the input data. so you do something like:

- passing it though layers:

`x = F.relu(self.something(x))`

(and the activation function)

and perfhaps second layer
`x = self.something_2(x)`

and dont forget to return the x

you never call forward directly. do:
`output = model(input)`

`nn.Linear` is the fully connected layer

### training

1. define the optimizer

`optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`

the adam is like a tuned up gradien descend wich the step = 0.01

2. define the loss function like the MSE by

`loss_fn = nn.MSELoss()`

### write the training loop

you do the hole thing quite manually. do the

1. clear stuff from previous training like
`optimizer.zero_grad()`
2. forward pass like
`outputs = model(X)`
3. compute loss
`loss = loss_fn(outputs, y)`
4. do the backward thingy
`loss.backward()`
5. update weights
`optimizer.step()`

and idk print stuff like the loss `loss.item()`