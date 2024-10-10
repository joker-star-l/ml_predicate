from numpy cimport intp_t, float64_t
from sklearn.tree cimport _tree

cdef class Node:

    def __cinit__(
        self,
        intp_t parent,
        bint is_left,
        bint is_leaf,
        intp_t feature,
        float64_t threshold,
        float64_t impurity,
        intp_t n_node_samples,
        float64_t weighted_n_node_samples,
        unsigned char missing_go_to_left,
        float64_t value
    ):
        self.parent = parent
        self.is_left = is_left
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples
        self.missing_go_to_left = missing_go_to_left
        self.value = value

def init_tree(_tree.Tree tree, intp_t capacity, intp_t max_depth, Node[:] nodes):
    tree.max_depth = max_depth
    tree._resize(capacity)
    for i in range(capacity):        
        # https://github.com/scikit-learn/scikit-learn/blob/1.5.1/sklearn/tree/_tree.pyx#L928C5-L932C60
        # cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
        #                   intp_t feature, float64_t threshold, float64_t impurity,
        #                   intp_t n_node_samples,
        #                   float64_t weighted_n_node_samples,
        #                   unsigned char missing_go_to_left)
        tree._add_node(
            nodes[i].parent,
            nodes[i].is_left,
            nodes[i].is_leaf,
            nodes[i].feature,
            nodes[i].threshold,
            nodes[i].impurity,
            nodes[i].n_node_samples,
            nodes[i].weighted_n_node_samples,
            nodes[i].missing_go_to_left
        )
        tree.value[i] = nodes[i].value
