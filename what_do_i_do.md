# hi

this is my bachlor diary

here i am going to informally keep track of stuff i learn, do, swear about and so on.

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

## OCGIN for dummies

- do a usual gnn
- do a sum of all nodes and you get a vector. (assuming every node representation was also a vector of each of nodes attributes). Formally: node emdebbing -> graph embedding
- so just a fucking sum. (for GIN). sometimes people use max/mean
- ah yeah, and this sum is taken from ALL layers (individually) so you have a sum vektor for every layer and then you concatenate them (stack on top of one another)
- aand that you value for the graph!

you can do an anomaly score. that would be just the difference between the calculated vector and the learned normal space center square

the loss is then this quadratic distance devided by N

while training we will minimize loss.

and after training we will compute the radius of the normal space. there you can make a choice:
- 95 quantile of normal distances
- mean +k* std
- or some other stuff

### and what about the center??
the center of the normal space is computer as the 1 step of training

after we have initilized random weights we pass all normal graps though the model, average the result and this is the center. now it is frozen forever.

### what bout node level?

do the same stuff but dont sum across the layer. 

there you can have either one center or a couple of centers for different kinds of nodes. but basically it is the same thing but without aggregation.


# how the fuck can i use logical constrains??

## usual nn

1. form logical constrains
2. turn logic into loss by saying
`L_total = L_data + lambda * L_logic`
3. turn your logic into soft logic

types of contrains:
- *soft* linda like penalties. can be violated per cost
- *hard* can never be violated

### example

logic is: A->B or not(A) + B

also translated as "A is high and B is low then penal"

example is in `pytorch_basics/doing_logical-constrains_with_nn.py`

there the `logits` are the raw outputs of the neural network

## building logical constrains for graphs

all info under `models/graph_decision_trees/readme.md`

one downside: can only be implemented for the first layer of the nn

# ocgin specifics

here i am just trying to undestand every little detail about an ocgin implementation in `models/simple_ocgin/simple_ocgin.py`

i got this piece of code by cutting around the Graph-Level-Anomaly detection library

### mlps?

as i have written an OCGIN is quite like a usual GNN but with an extra step. it is not quite true

an OCGIN consist of GIN layers, which consist of two steps:

1) (1 + lambda) * weight_v^(k - 1) # the weight of the previous layer
2) + BIG_SUM(weight_neightbors^(k -1)) # the sum of weight of the neighbors of the previous layer
3) sum 1 + 2. now you got a vector for sum of 1+ 2 for every of the node features
4) apply MLP. which stands for multi layer perceptron

which is pretty much a simple feed-forward neural network

here:
```
mlp = nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.ReLU(),
    nn.Linear(d_hidden, d_out),
)
```

the difference to a usual GNN is that the ususal GNN would apply to 1 + 2 something like a relu and be ready. a GIN uses a small network for every node.

in our example if every node has 7 features, then it would get for the first layer for node 1 a vector of length 7 as aggregated values of neightbors and then apply to this vector a simple neural network (which an output vector of whatever length we want. usually of the nuber of hidden dim)

by that a network can learn different non linear interpretations of every node, every feature

### self.register_buffer("center", torch.zeros(hidden_dim))

register_buffer stores non trainable model state

in this line we just kinda say, hey, i have this parameter, store it.

afterwards the center is initiated in `init_center`

### z = global_mean_pool(x, batch)

takes a mean of all states of the GIN (of quasi every node-embedding from all layers)

z.shape = [batch_size, hidden_dim]

batch size cuz we have a bunchof graphs in a batch.
so it is like one vector per graph

### init_center

`@torch.no_grad()` is a decorator which tells pytorch not to build a computation graph for everything inside this function

computation graph? -> see next subchater

`self.eval()` puts the model ito the evaluation mode, because we want deterministic embeddings, no randomness, no training time

evaluation mode? -> see next subchapter

n_samples - how many raphs we have proceeded

`torch.zeros_like` - create a vector/matrix of zeros and dimentions as the self.center

important: MLP parameters are randomly initialized

since `z.shape = [batch_size, hidden_dim]`

by `z.sum(dim=0)` we sum over batch_size and get a vector of hidden_dim

and z.size(0) gets the number of graphs in the batch

`self.center.copy_(center / n_samples)` updates center

### train_ocgin

`model.train()` sets model into the training mode

`model.parameters()` all trainable parameter (so not the center)

`lr` - learning rate

`optimizer.zero_grad()` stes gradients to zero
cuz in `loss.backward()` gradients are summed, not replaced.

`optimizer.step()` - updates the parameters with the gradients which are stores in the computation graph after applying loss.backward() 

### torch computation graph

torch kinda saves all operations we do.

like: Linear > ReLu > Linear > GINConv etc

uses it to compute gradients and stuff. and it saves tensors too. so it is computationally expensive and we need it only for training

### states of model

Some layers behave differently during training vs evaluation

(not in this particular model)

thats why a model has a training and evaluation mode.

e.g. Dropout > randomply zero some etries during training
would not do that in evaluation mode

a model has more states > TODO

### after training

now as we have trained the model, we can freely chose a boundary of the hypershere

it can be something like "take a 95 procent quantile of all ambeddings of all normal agraphs" and we get a hidden_dim vector of distances to the center.

after forwarding a new graph we take a mean of all node embeddings and see if it is in the boundary

the `compute_anomaly_scores` does exactly that. it computes the kinda distance squared from every graph to the center. if we would choose a boundary (95% quantile) we would take it from the anomaly scores

it we would test, we would do:

```
train_scores = compute_anomaly_scores(model, train_loader)
R = torch.quantile(train_scores, 0.95)

test_scores = compute_anomaly_scores(model, test_loader)
pred = (test_scores > R).int()  # 1 = anomaly, 0 = normal

```

# now guess i will start experimentin

## ablauf

first of all i can imagine i will have a bunch of models which i will want to compare to each other.

so it makes sence to write a module which would run stuff from different datasets and be like: run model1 and model2 with the same parameters and print results

the milestone is that i have no idea how different these models will be.

i can imagine those will be all OCGIN/OCGTL but i still havent understood how OCGTLs work.. and i feel now its not the time

so: in the models in every model folder i will have the model itself and its training loop

in the folder experiments i will have some kind of config where i would set:

- dataset
- number of layers
- learning rate
- hidden layers
- and the model

more about the exact implementation of experiments in experiments/readme.md