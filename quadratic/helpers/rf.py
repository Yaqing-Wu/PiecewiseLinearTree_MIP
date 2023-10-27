import numpy as np

# rf helper functions except for split_values_new & C_new are from https://github.com/ivanlin9522/TREE
def total_split_variable(trees_given): #this function returns the set of independent variables that are used in split conditions
    feature_set=set([])
    for tree in trees_given:
        feature=tree.tree_.feature[tree.tree_.feature>=0]
        feature_set=feature_set|set(feature)
    return feature_set


def V(trees_given,t,s): #this function returns variable that participates in split s
    tree=trees_given[t].tree_
    feature=tree.feature[s]
    return feature


def K(trees_given,i): #number of unique split points
    return split_values(trees_given,i).shape[0]


def get_input(forest): #this function captures the input tree ensemble and return list of trees embedded
    trees=list()
    for i in range(forest.n_estimators): 
        trees.append(forest.estimators_[i])
    return trees


def leaves(trees_given,t):
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==True]


def right_leaf(trees_given,t,s):  #return a list of all the right leaf of tree t, node s
    right_leaves=[]
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    stack = [s]
    node_id = stack.pop()
    if (children_left[node_id] != children_right[node_id]):
        stack.append(children_right[node_id])
    else:
        return right_leaves
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            right_leaves.append(node_id)
    return right_leaves


def left_leaf(trees_given,t,s): #return a list of all the left leaf of tree t, node s
    left_leaves=[]
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    stack = [s]
    node_id = stack.pop()
    if (children_left[node_id] != children_right[node_id]):
        stack.append(children_left[node_id])
    else:
        return left_leaves
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            left_leaves.append(node_id)
    return left_leaves

def prediction(trees_given,t,l,flag): #prediction of tree t, leaf l
    tree=trees_given[t].tree_
    if flag==0: #this is a classification tree and return which class the leaf predicts
        return np.argmax(tree.value[l,0,:])
    else: 
        return tree.value[:,0,0][l] #this is a regression tree 
    

def splits(trees_given,t): #return an array of splits(not leaf) of tree
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==False]


def is_it_leaf(trees_given,t): #this function returns an array of the boolean value, telling if it is leaf of the tree t
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [0]  # seed is the root node id
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True
    return is_leaves


def split_values(trees_given,i): #this function returns array of unique split points in ascendng order
    values=np.array([])
    if i in total_split_variable(trees_given): 
        for tree in trees_given:
            feature=set(tree.tree_.feature[tree.tree_.feature>=0])
            if i in feature:
                values=np.append(values,tree.tree_.threshold[tree.tree_.feature==i])
    values=np.unique(np.sort(values))
    return values


def C(trees_given,t,s): #set of values of variables i that participate in split
    #the expression in the paper is not right, since there is only one threshold in each split
    threshold=trees_given[t].tree_.threshold[s]
    feature=V(trees_given,t,s)
    return int(np.where(split_values(trees_given,feature)==threshold)[0])


def split_values_new(trees_given): # modified version of the split_values function
    
    value_dict = dict()
    values=np.array([])
    features = total_split_variable(trees_given)
    
    for feature in features:
        value_dict[feature] = np.array([])
        for tree in trees_given:
            feature_tree = set(tree.tree_.feature[tree.tree_.feature>=0])
            if feature in feature_tree:
                value_dict[feature] = np.append(value_dict[feature],tree.tree_.threshold[tree.tree_.feature==feature])
        value_dict[feature] = np.unique(np.sort(value_dict[feature]))
    
    return value_dict


def C_new(value_dict, trees_given, t, s): # modified version of the C funtion
    threshold=trees_given[t].tree_.threshold[s]
    feature=V(trees_given,t,s)
    value = int(np.where(value_dict[feature] == threshold)[0])
    return value