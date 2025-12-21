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

## GNNs in pytorch

instead of `nn.linear` Gnns use message passing layers like:

- `GCNConv`
- `GraphSAGEConv`
- `GATConv`
- `GINConv` - for OCGIN

these layers aggregate information from neighbors unlike normal layers

in practical terms change the layers and the forward function, since you need to pass the edge_index:

`x = self.conv1(x, edge_index)`

- you can also do like global mean pooling:
Take the mean of each feature across all nodes.
graph_emb = global_mean_pool(x, batch)