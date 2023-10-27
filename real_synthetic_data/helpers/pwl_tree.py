from sklearn.tree import _tree
import numpy as np
from collections import defaultdict


def tree_to_code(tree, feature_names):
    # obtain leaf node information for a trained tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto=dict()
    pathfinal = dict()
    predictions = dict()

    global k
    k = 0
    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            key = "{} <=".format(name)
            if node == 0:
                pathto[node] = {}
                pathto[node][key] = threshold
            else:
                pathto[node] = pathto[parent].copy()
                if key not in pathto[node]:
                    pathto[node][key] = threshold
                else:
                    if threshold < pathto[node][key]:
                        pathto[node][key] = threshold

            recurse(tree_.children_left[node], depth + 1, node)
            key = "{} >".format(name)
            if node == 0:
                pathto[node] = {}
                pathto[node][key] = threshold
            else:
                pathto[node] = pathto[parent].copy()
                if key not in pathto[node]:
                    pathto[node][key] = threshold
                else:
                    if threshold > pathto[node][key]:
                        pathto[node][key] = threshold
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            pathfinal[k] = pathto[parent]
            predictions[k] = tree_.value[node][0][0]
            k=k+1
        
        return pathfinal, predictions
    
    return recurse(0, 1, 0)


def get_coef(best_tree):
    # obtain coefficients and intercepts for leaf nodes
    linear_coeff = []
    linear_intercept = []
    for estimator in best_tree.estimators_:
        linear_coeff.append(estimator.coef_)
        linear_intercept.append(estimator.intercept_)
    linear_coeff = np.array(linear_coeff)
    linear_intercept = np.array(linear_intercept)
    linear_coeff = np.concatenate((linear_coeff, linear_intercept.reshape(-1,1)), axis = 1)
    return linear_coeff


def get_subdomain(dctree, n):
    # linear regression model for each leaf node
    tree_dict, _ = tree_to_code(dctree, feature_names=['x' + str(i) for i in range(n)])
    
    # obtain the info of the subdomain corresponding to the leaf
    # map the key from tree 'x_i >= value' & 'x_i <= value' to column number in the tree_params 2D array
    # tree_params: each row represents a leaf; the first (last) n column is the upper bound (lower bound) of each x_i (i in n)
    dict_names = {}
    count = 0 
    for i in range(n):
        lessthan = 'x' + str(i) + ' <='
        largerthan = 'x' + str(i) + ' >'
        dict_names[lessthan] = count
        dict_names[largerthan] = count + (n)
        count += 1

    n_leaves = len(tree_dict)
    
    tree_params = []
    tree_params1 = [1.0 for i in range(n)]
    tree_params2 = [0.0 for i in range(n)]
    tree_params = [tree_params1 + tree_params2]
    tree_params = np.repeat(tree_params, n_leaves, axis = 0)

    count = 0
    for _, leaf in tree_dict.items():
        for key, value in leaf.items():
            tree_params[count][dict_names[key]] = value        
        count += 1
    
    tree_params = np.asarray(tree_params)
        
    return tree_dict, tree_params


def y_min_max_calc(model, X, y_predict):
    # obtain the min/max y_predict values corresponding to each leaf
    
    # map each x value to the assigned leaf
    list_bin = model.transform_bins(X)
    
    # obtain predicted y values assigned to each leaf
    y_in_bins = defaultdict(list)
    for i in range(len(list_bin)):
        y_in_bins[list_bin[i]].append(y_predict[i])
    
    # ymin_in_bins (ymax_in_bins) contains min/max predicted y values for each leaf
    ymin_in_bins = []
    ymax_in_bins = []
    for i in range(int(max(list_bin)) + 1):
        ymin_in_bins.append(min(y_in_bins[i]))
        ymax_in_bins.append(max(y_in_bins[i]))

    return ymin_in_bins, ymax_in_bins

