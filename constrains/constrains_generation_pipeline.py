from constrains.graph_decision_trees.graph_level.train_and_print import train_from_node_level_features
from constrains.graph_decision_trees.node_level.train_and_print import train_for_model



def generate_constrains_gl(tree_loader, dataset_name, config):
    # extract tree params
    decision_tree_att_list = config["decision_tree"]["attribute_list"]
    decision_tree_max_depth = config["decision_tree"]["max_depth"]
    
    # split data into continious and categorical features
    
    # TODO
    
    # train desision tree on continius features
    filepath, tree_rules = train_from_node_level_features(
            tree_loader,
            decision_tree_att_list, 
            decision_tree_max_depth, 
            dataset_name
        )

    # train the logix model on vategorical features
    
    # merge the rules
    
    # save the merged rules as json
    
    return filepath


def generate_constrains_nl(data, tree_mask, dataset_name, config):
    # extract tree params
    decision_tree_att_list = config["decision_tree"]["attribute_list"]
    decision_tree_max_depth = config["decision_tree"]["max_depth"]
    
    # split data into continious and categorical features
    
    # TODO
    
    # train desision tree on continius features
    filepath, tree_rules = train_for_model(
            data,
            tree_mask,
            decision_tree_att_list, 
            decision_tree_max_depth, 
            dataset_name
        )

    # train the logix model on vategorical features
    
    # merge the rules
    
    # save the merged rules as json
    
    return filepath
