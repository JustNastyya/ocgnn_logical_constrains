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