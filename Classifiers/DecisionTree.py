import numpy as np
from collections import Counter


# store info about tree node
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        if self.value is not None:
            return True
        else:
            return False


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    #
    def fit(self, x, y):
        if self.num_features is None:
            self.num_features = x.shape[1]
        else:
            self.num_features = np.min(self.num_features, x.shape[1])

        self.root = self.expand_tree(x, y)

    def expand_tree(self, x, y, tree_depth=0):
        num_samples, num_features = x.shape
        num_labels = len(np.unique(y))

        # stopping criterion
        if (tree_depth >= self.max_depth) or (num_labels == 1) or (num_samples < self.min_samples_split):
            leaf_node_val = most_common_label(y)
            return TreeNode(value=leaf_node_val)  # returning the leaf node

        feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        # perform greedy search
        best_feature, best_threshold = calculate_best_criteria(x, y, feature_indices)
        left_indcs, right_indcs = split_data(x_col=x[:, best_feature], split_threshold=best_threshold)

        left = self.expand_tree(x=x[left_indcs, :], y=y[left_indcs], tree_depth=tree_depth + 1)
        right = self.expand_tree(x=x[right_indcs, :], y=y[right_indcs], tree_depth=tree_depth + 1)

        return TreeNode(best_feature, best_threshold, left, right)

    # traverse the tree and classify
    def predict(self, x):
        predictions = []
        for i in x:
            predictions.append(self.traverse_tree(i, self.root))

    def traverse_tree(self, sample, node):
        if node.is_leaf_node():
            return node.value

        if sample[node.feature] <= node.threshold:
            return self.traverse_tree(sample, node.left)
        return self.traverse_tree(sample, node.right)


# HELPER METHODS

# prob is the probability x/N
# - Sum ( prob(X) * log2( prob(X) ) )
def calculate_entropy(vect_y):
    histogram = np.bincount(vect_y)
    calc_val = []
    for i in histogram:
        probability = i / len(vect_y)
        if probability > 0:
            calc_val.append(probability * np.log2(probability))
    entropy = -np.sum(calc_val)
    return entropy


def most_common_label(y):
    counter = Counter(y)
    most_common_lbl = counter.most_common(1)[0][0]
    return most_common_lbl


def split_data(x_col, split_threshold):
    left_child_indcs = []
    right_child_indcs = []

    for i, val in zip(range(len(x_col)), x_col):
        if val <= split_threshold:
            left_child_indcs.append(i)
        else:
            right_child_indcs.append(i)

    return left_child_indcs, right_child_indcs


# information gain = entropy(parent_node) - [(prob(left_child) * entropy(left_chilf)) + (prob(right_child) *
# entropy(right_child)))]
def calculate_information_gain(y, x_col, threshold):
    # parent entropy
    parent_entropy = calculate_entropy(y)

    left_child, right_child = split_data(x_col, threshold)  # get the indices of the split

    if len(left_child) == 0 or len(right_child) == 0:
        return 0

    tot_len = len(y)
    left_child_len = len(left_child)
    right_child_len = len(right_child)

    left_child_entropy = calculate_entropy(y[left_child])
    right_child_entropy = calculate_entropy(y[right_child])

    overall_entropy = ((left_child_len / tot_len) * left_child_entropy) + \
                      ((right_child_len / tot_len) * right_child_entropy)

    info_gain = parent_entropy - overall_entropy

    return info_gain


# calculates the best criteria (feature and the corresponding threshold) for split
def calculate_best_criteria(x, y, feature_indices):
    best_gain = -1
    split_idx, split_threshold = None, None
    for feature_idx in feature_indices:
        x_col = x[:, feature_idx]
        thresholds = np.unique(x_col)
        for thrshld_val in thresholds:
            info_gain = calculate_information_gain(y, x_col, thrshld_val)

            if info_gain > best_gain:
                best_gain = info_gain
                split_idx = feature_idx
                split_threshold = thrshld_val

    return split_idx, split_threshold


if __name__ == "__main__":
    pass
