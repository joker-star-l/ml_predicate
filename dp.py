import onnx
import time
import pandas as pd
from typing import List, Tuple, Dict
import argparse
from tree import Node, TreeEnsembleRegressor, model2tree
import sys

# dp_v6: remove dp, only merge (more general scenarios). although calls dp, no dp here!!!

parser = argparse.ArgumentParser()
# test case: nyc-taxi-green-dec-2016_d10_l481_n961_20241010111047
# test case: house_16H_d5_l24_n47_20241108023539
# test case: bank-marketing_d10_l407_n813_20241106042040
# test case: NASA_d10_l264_n527_20241106064301
parser.add_argument('--model', '-m', type=str, default='Ailerons_d10_l818_n1635_20241112182051')
args = parser.parse_args()

model_name = args.model

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'

model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

# each not-leaf node except the end node in the chain has one leaf child, and these leaf children have a same target.
# the end node has a leaf child which also has the same target.
class MergeChain:
    def __init__(self, start_node: Node, end_node: Node, value: int):
        self.start_node = start_node
        self.end_node = end_node
        self.value = value

    def left_leaf_value(self, node: Node):
        return node.left.mode == b'LEAF' and int(node.left.target_weight) == self.value

    def has_same_feature(self):
        features = set()
        node = self.end_node
        while node != self.start_node.parent:
            if node.feature_id in features:
                return True
            features.add(node.feature_id)
            node = node.parent
        return False

    def merge(self):
        # key: (feature_id, left_leaf_value)
        node_map: Dict[Tuple[int, int], Node] = {}
        node = self.start_node
        while True:
            feature_id = node.feature_id
            left_leaf_value = self.left_leaf_value(node)
            ancestor_node = node_map.get((feature_id, left_leaf_value))
            if ancestor_node is None:  # not in the chain
                node_map[(feature_id, left_leaf_value)] = node
            else:  # merge the node to the ancestor node
                ancestor_node.value = node.value

                parent = node.parent
                if left_leaf_value:
                    ancestor_node.left.samples += node.left.samples

                    if parent.left == node:
                        parent.left = node.right
                    else:
                        parent.right = node.right
                    node.right.parent = parent
                else:
                    ancestor_node.right.samples += node.right.samples

                    if parent.left == node:
                        parent.left = node.left
                    else:
                        parent.right = node.left
                    node.left.parent = parent

                if node == self.end_node:
                    self.end_node = parent
                    break

            if node == self.end_node:
                break
            if left_leaf_value:
                node = node.right
            else:
                node = node.left
        
        self.update_samples()
    
    def update_samples(self):
        node = self.end_node
        while True:
            node.samples = node.left.samples + node.right.samples
            if node == self.start_node:
                break
            node = node.parent

    def print(self):
        ret = ''
        node = self.end_node
        while node != self.start_node.parent:
            s = f'{MergeChain.node_str(node.left)}, {MergeChain.node_str(node.right)}'
            ret = f'{s}\n{ret}'
            node = node.parent
        ret = f'{MergeChain.node_str(self.start_node)}\n{ret}'

        print(ret)
    
    @staticmethod
    def node_str(node: Node):
        if node.mode == b'LEAF':
            return f'[LEAF: {int(node.target_weight)}]'
        return f'[x{node.feature_id} <= {node.value:.6f}]'
    
def find_merge_chains(node: Node, merge_chains: List[MergeChain]):
    end_node, chain_value = find_merge_chains_(node, merge_chains)
    if chain_value in [0, 1]:
        merge_chains.append(MergeChain(node, end_node, chain_value))

def find_merge_chains_(node: Node, merge_chains: List[MergeChain]) -> Tuple[Node | None, int]:
    # chain_value:
    #  0 (0): node has 1 leaf child, and the target is 0
    #  1 (1): node has 1 leaf child, and the target is 1
    #  2 (0 or 1): node has 2 leaf children
    #  3 (0 and 1): node has 0 leaf child
    # return: chain_end_node, chain_value

    if node.left.mode == b'LEAF' and node.right.mode == b'LEAF':
        return node, 2

    if node.left.mode == b'LEAF':
        chain_value = int(node.left.target_weight)
        end_node, right_chain_value = find_merge_chains_(node.right, merge_chains)

        if right_chain_value == 2:  # 2 (0 or 1)
            return end_node, chain_value

        if right_chain_value == chain_value:  # 0 (0), 1 (1)
            return end_node, chain_value

        # 3 (0 and 1)
        return node, chain_value

    if node.right.mode == b'LEAF':
        chain_value = int(node.right.target_weight)
        end_node, left_chain_value = find_merge_chains_(node.left, merge_chains)

        if left_chain_value == 2:  # 2 (0 or 1)
            return end_node, chain_value

        if left_chain_value == chain_value:  # 0 (0), 1 (1)
            return end_node, chain_value

        # 3 (0 and 1)
        return node, chain_value

    left_end_node, left_chain_value = find_merge_chains_(node.left, merge_chains)
    if left_chain_value in [0, 1] and left_end_node != node.left:
        merge_chains.append(MergeChain(node.left, left_end_node, left_chain_value))

    right_end_node, right_chain_value = find_merge_chains_(node.right, merge_chains)
    if right_chain_value in [0, 1] and right_end_node != node.right:
        merge_chains.append(MergeChain(node.right, right_end_node, right_chain_value))

    return None, 3

start = time.perf_counter()
root = model2tree(model, samples_list, 0, None)

reduced_cost = 0

merge_chains: List[MergeChain] = []
find_merge_chains(root, merge_chains)
for i, merge_chain in enumerate(merge_chains):
    if merge_chain.has_same_feature():
        print(i)
        merge_chain.print()
        reduced_cost += merge_chain.start_node.branch_samples()
        merge_chain.merge()
        merge_chain.print()
        reduced_cost -= merge_chain.start_node.branch_samples()

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
