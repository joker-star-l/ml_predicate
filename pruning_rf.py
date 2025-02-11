import onnx
import joblib
from onnx import helper
import onnx.checker
import argparse
import pandas as pd
import time
from typing import List, Tuple

from utils import get_attribute
from tree import get_tree_intervals

start__ = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_t3_d2_l4_n7_20250204160726')
parser.add_argument('--predicate', '-p', type=float, default=1.275)
parser.add_argument('--clf2reg', action='store_true', default=False)
args = parser.parse_args()

# 回归树node attributes
# n_targets
# nodes_falsenodeids: 右侧分支
# nodes_featureids: 特征id
# nodes_hitrates
# nodes_missing_value_tracks_true
# nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
# nodes_nodeids
# nodes_treeids
# nodes_truenodeids: 左侧分支
# nodes_values: 阈值，叶子节点的值为0
# post_transform
# target_ids
# target_nodeids: 叶子节点的id
# target_treeids
# target_weights: 叶子节点的权重，即预测值

# 异常值处理?

# 分类树node attributes
# class_ids: 叶子节点权重对应的类别id
# class_nodeids: 叶子节点权重对应的节点id
# class_treeids: 叶子节点权重对应的树id
# class_weights: 叶子节点权重，即预测值
# classlabels_int64s: 类别id
# nodes_falsenodeids: 右侧分支
# nodes_featureids: 特征id
# nodes_hitrates
# nodes_missing_value_tracks_true
# nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
# nodes_nodeids
# nodes_treeids
# nodes_truenodeids: 左侧分支
# nodes_values: 阈值，叶子节点的值为0
# post_transform


if args.clf2reg:
    model_path = f'rf_model_output/{args.model}_reg'
else:
    model_path = f'rf_model/{args.model}'

out_path = f'rf_model_output/{args.model}_out'

if args.clf2reg:
    func = lambda x, n: int(x) == int(args.predicate)
else:
    func = lambda x, n: x > args.predicate / n


def pruning(tree_no, tree_interval, node_id, depth, result_nodes, onnx_model, f, tree_count) -> int:  # 0: leaf_false, 1: leaf_true, 2: inner
    tree_start = tree_interval[0]
    tree_end = tree_interval[1]

    left_nodes = get_attribute(onnx_model, 'nodes_truenodeids').ints[tree_start:tree_end]
    right_nodes = get_attribute(onnx_model, 'nodes_falsenodeids').ints[tree_start:tree_end]
    node_types = get_attribute(onnx_model, 'nodes_modes').strings[tree_start:tree_end]

    target_treeids = get_attribute(onnx_model, 'target_treeids').ints
    target_nodeids = get_attribute(onnx_model, 'target_nodeids').ints
    target_weights = get_attribute(onnx_model, 'target_weights').floats

    result_nodes[node_id] = node_types[node_id].decode('utf-8')
    is_leaf = node_types[node_id] == b'LEAF'

    if is_leaf:
        target_idx = -1
        for ti, ni in enumerate(target_nodeids):
            if ni == node_id and target_treeids[ti] == tree_no:
                target_idx = ti
                break

        result = int(f(target_weights[target_idx], tree_count))
        result_nodes[node_id] = 'LEAF_TRUE' if result == 1 else 'LEAF_FALSE'

        # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {result}')

        return result

    if not is_leaf:
        left_node_id = left_nodes[node_id]
        left_result = pruning(tree_no, tree_interval, left_node_id, depth + 1, result_nodes, onnx_model, f, tree_count)
        right_node_id = right_nodes[node_id]
        right_result = pruning(tree_no, tree_interval, right_node_id, depth + 1, result_nodes, onnx_model, f, tree_count)

        if left_result == 0 and right_result == 0:
            # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {0}')
            result_nodes[node_id] = 'LEAF_FALSE'
            result_nodes[left_node_id] = 'REMOVED'
            result_nodes[right_node_id] = 'REMOVED'
            return 0

        if left_result == 1 and right_result == 1:
            # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {1}')
            result_nodes[node_id] = 'LEAF_TRUE'
            result_nodes[left_node_id] = 'REMOVED'
            result_nodes[right_node_id] = 'REMOVED'
            return 1

        # print(f'node_id: {node_id}, depth: {depth}, leaf_depth: {depth + 1}')
        return 2


model = onnx.load(model_path + '.onnx')
# print(model)

tree_intervals = get_tree_intervals(model)

result_nodes_list: List[List[str]] = []
for (i, (start, end)) in enumerate(tree_intervals):
    result_nodes_list.append([None] * (end - start))

# only for debug
all_node_count = 0
all_removed_count = 0
all_node_cost_weights = 0
all_reduced_cost_weights = 0
node_cost_weights_list = []

for tree_no, result_nodes in enumerate(result_nodes_list):
    tree_interval = tree_intervals[tree_no]
    pruning(tree_no, tree_interval, 0, 0, result_nodes, model, func, len(tree_intervals))

    # only for debug
    # print(result_nodes)
    removed_count = result_nodes.count('REMOVED')
    print("tree:", tree_no, "total nodes:", len(result_nodes), "removed nodes:", removed_count, removed_count / len(result_nodes))
    all_node_count += len(result_nodes)
    all_removed_count += removed_count

    # only for debug
    node_cost_weights = [int(i) for i in get_attribute(model, 'nodes_hitrates').floats[tree_interval[0]:tree_interval[1]]]
    # print(node_cost_weights)
    reduced_cost_weights = sum([node_cost_weights[i] for i, node in enumerate(result_nodes) if node == 'REMOVED'])
    print("tree:", tree_no, "total cost:", sum(node_cost_weights), "reduced cost:", reduced_cost_weights, f"performance: {sum(node_cost_weights) / (sum(node_cost_weights) - reduced_cost_weights)}x")
    all_node_cost_weights += sum(node_cost_weights)
    all_reduced_cost_weights += reduced_cost_weights
    node_cost_weights_list.append(node_cost_weights)

# only for debug
assert all_node_count == len(get_attribute(model, 'nodes_modes').strings)
print("all", "total nodes:", all_node_count, "removed nodes:", all_removed_count, all_removed_count / all_node_count)
print("all", "total cost:", all_node_cost_weights, "reduced cost:", all_reduced_cost_weights, f"performance: {all_node_cost_weights / (all_node_cost_weights - all_reduced_cost_weights)}x")


def reg2reg(input_model, removed_nodes_list: List[List[str]], tree_intervals: List[Tuple[int, int]]):    
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
    
    # attribute
    tree_leaf_counts = [removed_nodes.count('LEAF_FALSE') + removed_nodes.count('LEAF_TRUE') for removed_nodes in result_nodes_list]

    new_ids_list: List[Tuple[int, str]] = []
    for removed_nodes in removed_nodes_list:
        new_ids = []
        id_ = 0  # new id in a single tree
        for node in removed_nodes:
            if (node == 'LEAF_FALSE' or node == 'LEAF_TRUE'):
                new_ids.append([id_, node])
                id_ += 1
            elif node == 'BRANCH_LEQ':
                new_ids.append([id_, node])
                id_ += 1
            else:
                new_ids.append([-1, node])
        
        new_ids_list.append(new_ids)

    # # n_targets
    n_targets = input_n_targets

    # # nodes_falsenodeids: 右侧分支
    nodes_falsenodeids = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_flasenodeids = [(new_ids[ii][0] if new_ids[ii][0] != -1 else 0) for i , ii in enumerate(input_nodes_falsenodeids[tree_start:tree_end]) if new_ids[i][1] != 'REMOVED']
        nodes_falsenodeids.extend(tree_nodes_flasenodeids)

    # # nodes_featureids: 特征id
    nodes_featureids = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_featureids = [(ii if new_ids[i][1] == 'BRANCH_LEQ' else 0) for i, ii in enumerate(input_nodes_featureids[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_featureids.extend(tree_nodes_featureids)

    # # nodes_hitrates
    nodes_hitrates = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_hitrates = [ii for i, ii in enumerate(input_nodes_hitrates[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_hitrates.extend(tree_nodes_hitrates)

    # # nodes_missing_value_tracks_true
    nodes_missing_value_tracks_true = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_missing_value_tracks_true = [ii for i, ii in enumerate(input_nodes_missing_value_tracks_true[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_missing_value_tracks_true.extend(tree_nodes_missing_value_tracks_true)

    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    nodes_modes = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_modes = [('BRANCH_LEQ' if new_id[1] == 'BRANCH_LEQ' else 'LEAF') for new_id in new_ids if new_id[0] != -1]
        nodes_modes.extend(tree_nodes_modes)

    # # nodes_nodeids
    nodes_nodeids = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_nodeids = [new_ids[i][0] for i, _ in enumerate(input_nodes_nodeids[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_nodeids.extend(tree_nodes_nodeids)

    # # nodes_treeids
    nodes_treeids = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_treeids = [ii for i, ii in enumerate(input_nodes_treeids[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_treeids.extend(tree_nodes_treeids)

    # # nodes_truenodeids: 左侧分支
    nodes_truenodeids = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_truenodeids = [(new_ids[ii][0] if new_ids[ii][0] != -1 else 0) for i , ii in enumerate(input_nodes_truenodeids[tree_start:tree_end]) if new_ids[i][1] != 'REMOVED']
        nodes_truenodeids.extend(tree_nodes_truenodeids)

    # # nodes_values: 阈值，叶子节点的值为0
    nodes_values = []
    for tree_no, (tree_start, tree_end) in enumerate(tree_intervals):
        new_ids = new_ids_list[tree_no]
        tree_nodes_values = [(ii if new_ids[i][1] == 'BRANCH_LEQ' else 0) for i, ii in enumerate(input_nodes_values[tree_start:tree_end]) if new_ids[i][0] != -1]
        nodes_values.extend(tree_nodes_values)

    # # post_transform
    post_transform = input_post_transform

    # # target_ids
    target_ids = [0] * sum(tree_leaf_counts)

    # # target_nodeids: 叶子节点的id
    target_nodeids = []
    for new_ids in new_ids_list:
        tree_target_nodeids = [new_id[0] for new_id in new_ids if (new_id[1] == 'LEAF_FALSE' or new_id[1] == 'LEAF_TRUE')]
        target_nodeids.extend(tree_target_nodeids)

    # # target_treeids
    target_treeids = []
    for tree_no, tree_leaf_count in enumerate(tree_leaf_counts):
        tree_target_treeids = [tree_no] * tree_leaf_count
        target_treeids.extend(tree_target_treeids)

    # # target_weights: 叶子节点的权重，即预测值
    target_weights = []
    for new_ids in new_ids_list:
        tree_target_weights = [int(new_id[1] == 'LEAF_TRUE') / len(tree_intervals) for new_id in new_ids if (new_id[1] == 'LEAF_FALSE' or new_id[1] == 'LEAF_TRUE')]
        target_weights.extend(tree_target_weights)

    # node
    node = helper.make_node(
        op_type='TreeEnsembleRegressor',
        inputs=[input_model.graph.input[0].name],
        outputs=[input_model.graph.output[0].name],
        name='TreeEnsembleRegressor',
        domain='ai.onnx.ml',
        # attributes
        n_targets=n_targets,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        post_transform=post_transform,
        target_ids=target_ids,
        target_nodeids=target_nodeids,
        target_treeids=target_treeids,
        target_weights=target_weights
    )

    # graph
    graph = helper.make_graph(
        nodes=[node],
        name=input_model.graph.name,
        initializer=[],
        inputs=input_model.graph.input,
        outputs=input_model.graph.output,
    )
    
    # model
    output_model = helper.make_model(
        graph=graph,
        opset_imports=input_model.opset_import,
    )
    output_model.ir_version = input_model.ir_version

    onnx.checker.check_model(output_model)

    print("input features:", set(input_nodes_featureids), len(set(input_nodes_featureids)), "output features:", set(nodes_featureids), len(set(nodes_featureids)))

    return output_model

output_model = reg2reg(model, result_nodes_list, tree_intervals)
# print(output_model)

onnx.save_model(output_model, out_path + '.onnx')

# only for debug
node_samples = []
for tree_no, _ in enumerate(tree_intervals):
    result_nodes = result_nodes_list[tree_no]
    node_cost_weights = node_cost_weights_list[tree_no]
    for i, stat in enumerate(result_nodes):
        if stat != 'REMOVED':
            node_samples.append(node_cost_weights[i])

# only for debug
for i, f in enumerate(get_attribute(output_model, "nodes_hitrates").floats):
    if node_samples[i] != int(f):
        raise Exception(f"node_samples[{i}] != int(f), {node_samples[i]} != {int(f)}")

# only for debug
df = pd.DataFrame(node_samples, columns=['node_samples'])
df.to_csv(out_path + '_node_samples.csv', index=True)

end__ = time.time()

print("time: ", end__ - start__)
