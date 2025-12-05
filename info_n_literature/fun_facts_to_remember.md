## from anh or master thesis

### unsupervised approach
In contrast, when labelled data is limited, unsupervised approaches are often
preferred due to their flexibility and scalability. Unsupervised outlier detection methods are the
main focus of this study. These methods generally assume that normal objects form one big
group or multiple groups together, whereas outlier objects do not fit into any of the common
patterns of those groups. This allows the identification of hidden or unseen outlier patterns.

### kinds of outliers:
Node outliers: These are nodes that significantly deviate from others in terms of node-level
metrics, such as node degree or centrality. For example, in social networks, a node outlier
might be a user with an exceptionally high number of connections.
• Edge outliers: This refers to an unexpected connection between two nodes or an unusual
edge’s property. For instance, in a citation dataset, an edge outlier could be high-frequency
citations between two authors working in different organizations and domains.
• Subgraph outliers: This involves a subset of nodes, or a subgraph, behaving anomalously
when considered as a collection, even if each node and edge is viewed as normal. For
example, in an e-commerce dataset, bot users may tend to post positive reviews about
certain products and negative reviews about many others.
• Structural outliers: These are anomalies apparent only when considering the graph structure. For example, an outlier edge that connects two different communities in graphs.
• Contextual outliers: Nodes or edges that are normal in general but become anomalies
under specific contexts or conditions.
• Global outliers: Nodes with attributes significantly different from the overall pattern of
the entire graph.
• Community outliers: Nodes with attributes significantly different from their neighbourhood.
• Temporal outliers: In dynamic graphs like transportation networks, unusual events occurring at specific time points may be related to outlier nodes or edges.

### statistical methods:
This method makes use of nonnegative matrix factorization [10], involving the decomposition of
A into matrices U and V , such that A ≈ UV T
, to minimize the objective function ∥A − UV T ∥,
with constraints U ≥ 0 and V ≥ 0. The low-rank approximation UV T assumes the modeling of
normal data in G; meanwhile, the residual matrix R = A−UV T
serves as a strong indicator for
entries that do not adhere to the low-rank assumption. Therefore, the outlier scores for edges
are determined by the absolute values in matrix R.

### other statistical:
An alternative approach [12] utilizes the Minimum Description Length (MDL) and the principle
of the SUBDUE system to detect outlier subgraphs. SUBDUE [13] is a method of detecting
repetitive patterns in graphs. MDL is the smallest number of bits required to encode a piece
of data; SUBDUE approximates this value for any subgraph and finds the best substructures
to compress a graph