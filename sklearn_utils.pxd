from numpy cimport intp_t, float64_t

# https://github.com/scikit-learn/scikit-learn/blob/1.5.1/sklearn/tree/_tree.pyx#L928C5-L932C60
# cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
#                   intp_t feature, float64_t threshold, float64_t impurity,
#                   intp_t n_node_samples,
#                   float64_t weighted_n_node_samples,
#                   unsigned char missing_go_to_left)
cdef class Node:
    cdef public intp_t parent
    cdef public bint is_left
    cdef public bint is_leaf
    cdef public intp_t feature
    cdef public float64_t threshold
    cdef public float64_t impurity
    cdef public intp_t n_node_samples
    cdef public float64_t weighted_n_node_samples
    cdef public unsigned char missing_go_to_left
    cdef public float64_t value
