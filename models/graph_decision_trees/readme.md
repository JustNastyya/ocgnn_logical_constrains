# Graph decision trees

Main idea is to implement graph decision trees in order to generate simple logical constrains to use them later in OCGIN.

For every level a specialized model adn config are spezified.

the config is a class implementing feature extraction for every node

there a list of node features is to be defined

## graph level

tu run execute `train_and_print.py`.

You can specialize:

- dataset
- features in `attribute_list`
- max depth

available features:
- `node_degree`
- `clustering_coefficient`

some of the results are saved in `tud_results.md` 