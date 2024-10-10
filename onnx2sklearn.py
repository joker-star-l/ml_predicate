import onnx
from sklearn.tree import DecisionTreeRegressor, _tree
from utils import get_attribute
import time
import numpy as np
import joblib
from typing import List
import sklearn_utils as skutils
from tree import *
import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='Ailerons_d10_l703_n1405_20240915180213')
parser.add_argument('--step', '-s', type=int, default=0)
args = parser.parse_args()

model_name = args.model
if args.step == 0:
    sufix = '_out'
elif args.step == 1:
    sufix = '_out2'
else:
    raise ValueError('Invalid step')

sklearn_path = f'model/{model_name}.joblib'
onnx_path = f'model_output/{model_name}{sufix}.onnx'
output_path = onnx_path.replace('.onnx', '.joblib')
same_tree = onnx_path[:-5] == sklearn_path[:-7]

def get_max_depth(input_model: onnx.ModelProto) -> int:
    return get_max_depth_internal(input_model, 0) - 1

def get_max_depth_internal(input_model: onnx.ModelProto, idx: int) -> int:
    # input model attributes
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, 'nodes_truenodeids').ints

    left_depth = 1
    right_depth = 1
    if input_nodes_truenodeids[idx] != 0:
        left_depth += get_max_depth_internal(input_model, input_nodes_truenodeids[idx])
    if input_nodes_falsenodeids[idx] != 0:
        right_depth += get_max_depth_internal(input_model, input_nodes_falsenodeids[idx])
    return max(left_depth, right_depth)

def onnx2sklearn(input_model: onnx.ModelProto, model: DecisionTreeRegressor):
    # input model attributes
    # # n_targets
    input_n_targets = get_attribute(input_model, 'n_targets').i
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, 'nodes_hitrates').floats
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(input_model, 'nodes_missing_value_tracks_true').ints
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_node_modes = get_attribute(input_model, 'nodes_modes').strings
    # # nodes_nodeids
    input_nodes_nodeids = get_attribute(input_model, 'nodes_nodeids').ints
    # # nodes_treeids
    input_nodes_treeids = get_attribute(input_model, 'nodes_treeids').ints
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, 'nodes_truenodeids').ints
    # # nodes_values: 阈值，叶子节点的值为0
    input_nodes_values = get_attribute(input_model, 'nodes_values').floats
    # # post_transform
    input_post_transform = get_attribute(input_model, 'post_transform').s
    # # target_ids
    input_target_ids = get_attribute(input_model, 'target_ids').ints
    # # target_nodeids: 叶子节点的id
    input_target_nodeids = get_attribute(input_model, 'target_nodeids').ints
    # # target_treeids
    input_target_treeids = get_attribute(input_model, 'target_treeids').ints
    # # target_weights: 叶子节点的权重，即预测值
    input_target_weights = get_attribute(input_model, 'target_weights').floats

    input_depth = get_max_depth(input_model)
    print('depth:', input_depth)

    n_features = len(set(input_nodes_featureids))
    print('n_features:', n_features)

    node_count = len(input_nodes_nodeids)
    print('node_count:', node_count)

    tree = _tree.Tree(model.tree_.n_features, model.tree_.n_classes, model.tree_.n_outputs)

    root = model2tree(input_model, None, 0, None)
    nodes: List[Node] = []
    preorder(root, nodes)
    
    # only for debug
    for (i, node) in enumerate(nodes):
        if i != node.id:
            raise ValueError(f'node id not match: {i} != {node.id}')

    sknodes = np.ndarray(shape=node_count, dtype=skutils.Node)
    for (i, node) in enumerate(nodes):
        sknodes[i] = skutils.Node(
            _tree.TREE_UNDEFINED if node.parent is None else node.parent.id,
            node.parent is not None and node.parent.left == node,
            node.mode == b'LEAF',
            node.feature_id,
            node.value,
            0.0,
            node.samples,
            float(node.samples),
            False,
            0.0 if node.target_weight is None else node.target_weight
        )

    skutils.init_tree(tree, node_count, model.max_depth, sknodes)

    # only for debug when onnx and joblib model are the same
    if same_tree:
        if (model.tree_.children_left - tree.children_left).sum() != 0:
            raise ValueError('children_left not match')
        if (model.tree_.children_right - tree.children_right).sum() != 0:
            raise ValueError('children_right not match')
        if (model.tree_.feature - tree.feature).sum() != 0:
            raise ValueError('feature not match')
        th = (model.tree_.threshold - tree.threshold).sum()
        print('threshold diff:', th)
        if th > 1e-4:
            raise ValueError('threshold not match')
        if (model.tree_.n_node_samples - tree.n_node_samples).sum() != 0:
            raise ValueError('n_node_samples not match')
        if (model.tree_.weighted_n_node_samples - tree.weighted_n_node_samples).sum() != 0:
            raise ValueError('weighted_n_node_samples not match')

    model.tree_ = tree

start = time.perf_counter()

onnx_model = onnx.load(onnx_path)
sklearn_model = joblib.load(sklearn_path)

if same_tree:
    a = sklearn_model.predict([np.zeros(sklearn_model.tree_.n_features), np.ones(sklearn_model.tree_.n_features)])

onnx2sklearn(onnx_model, sklearn_model)
joblib.dump(sklearn_model, output_path)

if same_tree:
    b = sklearn_model.predict([np.zeros(sklearn_model.tree_.n_features), np.ones(sklearn_model.tree_.n_features)])
    print(a, b)
    if a.sum() - b.sum() > 1e-4:
        raise ValueError('onnx2sklearn failed')

end = time.perf_counter()

print(f'Running time: {end - start} Seconds')
