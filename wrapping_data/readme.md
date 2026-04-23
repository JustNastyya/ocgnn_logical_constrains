# experiments description:

i have:
- two constrain handlers
- 3 loss types
- two levels (node + graph)
- (possibly) two types of generating the constrains
- about 15 datasets
- about a 1000 hyperparameters combinations
- two types of models (with and without forecasting)

there are planned 4 big groups of experiments:

- lc by decision tree generation and no lc in forecasting   - done
- lc by decision tree generation and with lc in forecasting - partially done
- lc by binary nns and no lc in forecasting
- lc by binary nns and with lc in forecasting

used datasets:
node:
- CS            Coauthor authors that are connected by an edge if they co-authored a paper. 
- CiteSeer      small dataset citation network benchmark
- Cora          halt cora
- Cornell       the one about students of cornell
- Photo         from amazon
- PubMed        pubmed halt
- Texas         lp
- Wisconsin

graph:
- AIDS          small molecules
- COIL-RAG      computer vision
- ENZYMES       bioinformatics
- DHFR          small molecules
- MSRC_21       computer vision
- MUTAG         small molecules
- PROTEINS      bioinformatics


per dataset suposed number of experiments: 1952
without the 512th layer: 1708

# details?

here i shall put the statistical analysis of ran experiments

each folder - one new method
in the folder:
the script which generated the graphs and a tex file with the description of the thing

## which experiments where?

the pure loss based is the one where i have used only loss based methods from my presentation
(loss not used in prediction)

constrains forecasting is the one where i use the constrained value in anomaly score calculation


