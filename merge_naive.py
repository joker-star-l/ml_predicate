import onnx
import time
import pandas as pd
from typing import List, Tuple, Dict
import argparse
from tree import Node, TreeEnsembleRegressor, model2trees
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_t10_d10_l849_n1698_20250209144203')
parser.add_argument('--conservative', '-c', action='store_true', default=False)
args = parser.parse_args()

model_name = args.model
conservative = args.conservative

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'

model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

start = time.perf_counter()
roots = model2trees(model, samples_list)

# 两侧必须有紧挨着超平面的纯色区域才可以滑动
# 与当前节点特征不同的节点两侧都要遍历
# 与当前节点特征相同的节点只需要遍历靠近当前节点的那一侧
# 删除最靠近当前节点的同特征节点

M_FALSE = 0
M_TRUE = 1
M_NO = 2

def find_merge_nodes(node: Node, path_length: int, root: Node, left_branch: bool, result: List[Tuple[Node, int]]) -> int:
    if node.mode == b'LEAF':
        return M_FALSE if node.target_weight == 0 else M_TRUE
    
    same_feature = node.feature_id == root.feature_id

    # 特征不相同或特征相同但是 node 在 root 的右侧时, 需要递归遍历左子树
    if not same_feature or (same_feature and not left_branch):
        left_merge_stats = find_merge_nodes(node.left, path_length + 1, root, left_branch, result)
        if left_merge_stats == M_NO:
            return M_NO
    
    # 特征不相同或特征相同但是 node 在 root 的左侧时, 需要递归遍历右子树
    if not same_feature or (same_feature and left_branch):
        right_merge_stats = find_merge_nodes(node.right, path_length + 1, root, left_branch, result)
        if right_merge_stats == M_NO:
            return M_NO
    
    if not same_feature:
        # 特征不同, 左右两侧的状态不同, 无法合并
        if left_merge_stats != right_merge_stats:
            return M_NO
        else:
            return left_merge_stats
    
    if same_feature and not left_branch:
        if left_merge_stats != M_NO:
            update_result(node, path_length, root, result)
        return left_merge_stats
    
    if same_feature and left_branch:
        if right_merge_stats != M_NO:
            update_result(node, path_length, root, result)
        return right_merge_stats

def update_result(node: Node, path_length: int, root: Node, result: List[Tuple[Node, int]]):
    if node.mode == b'LEAF':
        return

    assert node.feature_id == root.feature_id
    
    if not result:
        result.append((node, path_length))
        return

    last = result[-1][0]

    assert node.feature_id == last.feature_id
    
    new_delta = abs(round(round(node.value, 6) - round(root.value, 6), 6))
    old_delta = abs(round(round(last.value, 6) - round(root.value, 6), 6))

    if new_delta == old_delta:
        result.append((node, path_length))
        return
    
    if new_delta < old_delta:
        result.clear()
        result.append((node, path_length))
        return

    return

def merge(root: Node, nodes: List[Node], left_branch: bool):
    # only for debug
    assert root.mode == b'BRANCH_LEQ'
    assert len(nodes) > 0
    for node in nodes:
        assert node.mode == b'BRANCH_LEQ'
        assert node.feature_id == root.feature_id

    # change root properties
    root.value = nodes[0].value

    # change links
    for node in nodes:
        if left_branch:
            parent = node.parent
            left = node.left

            # link
            if node == parent.left:
                parent.left = left
            else:
                parent.right = left
            left.parent = parent

            # only for debug
            # here just a mock, TODO change me
            delta = node.samples - left.samples
            curr = left
            while curr.mode != b'LEAF':
                curr = curr.left
            curr.samples += delta
            left.update_samples()

        else:
            parent = node.parent
            right = node.right
            
            # link
            if node == parent.left:
                parent.left = right
            else:
                parent.right = right
            right.parent = parent

            # only for debug
            # here just a mock, TODO change me
            delta = node.samples - right.samples
            curr = right
            while curr.mode != b'LEAF':
                curr = curr.right
            curr.samples += delta
            right.update_samples()

        # clear
        node.parent = None
        node.left = None
        node.right = None


def dfs(node: Node):
    # print(node.mode.decode(), 'feature_id:', node.feature_id)
    if node.mode == b'LEAF':
        return
    dfs(node.left)
    dfs(node.right)

    left_merge_nodes = []
    left_merge_stats = find_merge_nodes(node.left, 1, node, True, left_merge_nodes)
    if left_merge_stats == M_NO:
        print(node.id, "left cannot merge")
        return
    
    right_merge_nodes = []
    right_merge_stats = find_merge_nodes(node.right, 1, node, False, right_merge_nodes)
    if right_merge_stats == M_NO:
        print(node.id, "right cannot merge")
        return
    
    if left_merge_stats != right_merge_stats:
        print(node.id, "cannot merge")
        return

    print(node.id, "can merge!", "left_nodes", left_merge_nodes, "right_nodes", right_merge_nodes)

    max_left_path_length = max([path_length for (_, path_length) in left_merge_nodes], default=0)
    max_right_path_length = max([path_length for (_, path_length) in right_merge_nodes], default=0)
    left_merge_nodes = [node for (node, _) in left_merge_nodes]
    right_merge_nodes = [node for (node, _) in right_merge_nodes]

    if left_merge_nodes and right_merge_nodes:
        print(node.id, "can merge both sides!")
        global conservative
        if conservative:
            return
        if max_left_path_length > max_right_path_length:
            merge(node, left_merge_nodes, True)
        else:
            merge(node, right_merge_nodes, False)
        return
    
    if left_merge_nodes:
        merge(node, left_merge_nodes, True)
        return

    merge(node, right_merge_nodes, False)

for i, root in enumerate(roots):
    print("<tree>", i)
    dfs(root)

# only for debug
def debug_samples(root: 'Node') :
    if root.mode != b'LEAF':
        if root.samples != debug_samples(root.left) + debug_samples(root.right):
            raise ValueError('Samples not match')
    return root.samples

reduced_cost = 0

regressor = TreeEnsembleRegressor.from_trees(roots)
output_model = regressor.to_model(model)
onnx.save_model(output_model, model_path.replace('_out.onnx', '_out2.onnx'))
end = time.perf_counter()

print(f'Elapsed time: {end - start:.6f}s')

# only for debug
branch_samples = 0
for root in roots:
    branch_samples += root.branch_samples()
print(f'Branch samples: {branch_samples}', sum(samples_list))
reduced_cost = sum(samples_list) - branch_samples # TODO remove me
print(f'Reduced cost: {reduced_cost}')
if branch_samples + reduced_cost != sum(samples_list):
    raise ValueError('Branch samples not match')

# only for debug
for root in roots:
    debug_samples(root)

print(f'Performance: {sum(samples_list) / (sum(samples_list) - reduced_cost)}')
