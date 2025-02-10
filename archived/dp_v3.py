import onnx
import time
import pandas as pd
from typing import List, Tuple
import argparse
from tree import Node, TreeEnsembleRegressor, model2tree

# dp_v3: remove dp, only merge. although calls dp, no dp here!!!
# TODO: bug fix: some nodes have the same feature, but cannot be deleted

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='house_16H_d10_l280_n559_20241009120728')
args = parser.parse_args()

model_name = args.model

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'

model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

def get_leaf_nodes(node: 'Node', depth: int, leaf_nodes: List[Tuple['Node', int]]):
    # 左根右
    depth += 1
    if node.mode == b'LEAF':
        leaf_nodes.append((node, depth))
    else:
        get_leaf_nodes(node.left, depth, leaf_nodes)
        get_leaf_nodes(node.right, depth, leaf_nodes)

def get_can_merge_nodes(leaf_nodes: List[Tuple['Node', int]]) -> List[List[Tuple['Node', int]]]:
    # the inner list has three elements: common parent, node1 (shorter to parent), node2 (longer to parent)
    merge_nodes_list: List[List[Tuple['Node', int]]] = []
    
    i = 0
    for i in range(len(leaf_nodes) - 1):
        l1 = leaf_nodes[i]
        l2 = leaf_nodes[i + 1]

        if l1[0].target_weight != l2[0].target_weight:
            continue

        p1 = (l1[0].parent, l1[1] - 1)
        p2 = (l2[0].parent, l2[1] - 1)

        feature_id = p1[0].feature_id
        d1 = 1
        d2 = 1
        while p1[0] != p2[0]:
            if feature_id != p1[0].feature_id or feature_id != p2[0].feature_id:
                # cannot merge
                break

            if p1[1] > p2[1]:
                p1 = (p1[0].parent, p1[1] - 1)
                d1 += 1
            elif p2[1] > p1[1]:
                p2 = (p2[0].parent, p2[1] - 1)
                d2 += 1
            else:
                p1 = (p1[0].parent, p1[1] - 1)
                d1 += 1
                p2 = (p2[0].parent, p2[1] - 1)
                d2 += 1

        if p1[0] == p2[0] and feature_id == p1[0].feature_id:
            print(f'can merge: {i} {i + 1}, n_nodes: {2 * len(leaf_nodes) - 1}')

            if d1 <= d2:
                merge_nodes_list.append([p1, (l1[0], d1), (l2[0], d2)])
            else:
                merge_nodes_list.append([p1, (l2[0], d2), (l1[0], d1)])

    return merge_nodes_list

def merge(nodes: List[Tuple['Node', int]]) -> int:
    reduced_cost = 0
    
    common_parent = nodes[0][0]

    # the longer path node
    node = nodes[2][0]
    parent = node.parent
    reduced_cost += node.samples + parent.samples

    print('common_parent.value', common_parent.value, 'shorter_node_parent.value', nodes[1][0].parent.value, 'longer_node_parent.value', nodes[2][0].parent.value)

    # change common parent node threshold
    common_parent.value = parent.value
    print('common_parent.value_', common_parent.value)

    if parent.left == node:
        another = parent.right
        parent.right = None
    else:
        another = parent.left
        parent.left = None

    # change parent.parent to another, parent.parent always not null
    if parent.parent.left == parent:
        parent.parent.left = another
    else:
        parent.parent.right = another
    another.parent = parent.parent
    parent.parent = None

    # change longer path node samples
    parent = another.parent
    merge_samples = node.samples
    while parent != common_parent:
        parent.samples -= merge_samples
        parent = parent.parent

        reduced_cost += merge_samples

    # change shorter path node samples
    node = nodes[1][0]
    while node != common_parent:
        node.samples += merge_samples
        node = node.parent

        reduced_cost -= merge_samples

    return reduced_cost

start = time.perf_counter()
root = model2tree(model, samples_list, 0, None)

leaf_nodes: List[Tuple['Node', int]] = []
get_leaf_nodes(root, 0, leaf_nodes)
merge_nodes_list = get_can_merge_nodes(leaf_nodes)
merge_nodes_list.sort(key=lambda x: x[0][1], reverse=True)

reduced_cost = 0
for merge_nodes in merge_nodes_list:
    reduced_cost += merge(merge_nodes)

regressor = TreeEnsembleRegressor.from_tree(root)
output_model = regressor.to_model(model)
onnx.save_model(output_model, model_path.replace('_out.onnx', '_out2.onnx'))
end = time.perf_counter()

print(f'Elapsed time: {end - start:.6f}s')

print(f'Branch samples: {root.branch_samples()}', sum(samples_list))

print(f'Reduced cost: {reduced_cost}')

# only for debug
if root.branch_samples() + reduced_cost != sum(samples_list):
    raise ValueError('Branch samples not match')

# only for debug
def debug_samples(root: 'Node') :
    if root.mode != b'LEAF':
        if root.samples != debug_samples(root.left) + debug_samples(root.right):
            raise ValueError('Samples not match')
    return root.samples
debug_samples(root)

print(f'Performance: {sum(samples_list) / (sum(samples_list) - reduced_cost)}')
