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
# test case: nyc-taxi-green-dec-2016_d10_l858_n1715_20241010162144
# test case: nyc-taxi-green-dec-2016_d10_l859_n1717_20241128134842
# test case: house_16H_d5_l24_n47_20241108023539
# test case: bank-marketing_d10_l407_n813_20241106042040
# test case: NASA_d10_l264_n527_20241106064301
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l859_n1717_20241128134842')
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
        self.value = value  # only in [0, 1, 2, None]

    def left_leaf_value(self, node: Node):
        if self.value is None:
            return False
        
        return node.left.mode == b'LEAF' and int(node.left.target_weight) == self.value

    def has_same_feature(self):
        if self.value not in [0, 1]:
            return False

        features = set()
        node = self.end_node
        while node != self.start_node.parent:
            if node.feature_id in features:
                return True
            features.add(node.feature_id)
            node = node.parent
        return False

    # intra-chain merge
    def merge(self):
        if self.value not in [0, 1]:
            return

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

                print('intra-chain merging...')

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
        self.check_and_update_value()
    
    def update_samples(self):
        if self.value is None:
            return

        node = self.end_node
        while True:
            node.samples = node.left.samples + node.right.samples
            if node == self.start_node:
                break
            node = node.parent

    def print(self):
        if self.value is None:
            print()
            return

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
            return f'[LEAF: {int(node.target_weight)}, ({node.samples})]'
        return f'[x{node.feature_id} <= {node.value:.6f}, ({node.samples})]'
    
    def check_and_update_value(self):
        if self.start_node == self.end_node and self.end_node.left.mode == b'LEAF' and self.end_node.right.mode == b'LEAF':
            self.value = 2
    
def find_merge_chains(node: Node, merge_chains: List[MergeChain]):
    end_node, chain_value = find_merge_chains_(node, merge_chains)
    if chain_value in [0, 1, 2]:
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
    if left_chain_value in [0, 1, 2] and left_end_node != node.left:
        merge_chains.append(MergeChain(node.left, left_end_node, left_chain_value))

    right_end_node, right_chain_value = find_merge_chains_(node.right, merge_chains)
    if right_chain_value in [0, 1, 2] and right_end_node != node.right:
        merge_chains.append(MergeChain(node.right, right_end_node, right_chain_value))

    return None, 3

def list_to_parent_map(merge_chains: List[MergeChain]) -> Dict[Node, List[MergeChain]]:
    parent_map: Dict[Node, List[MergeChain]] = {}
    for chain in merge_chains:
        parent = chain.start_node.parent
        if parent is not None:
            if parent not in parent_map:
                parent_map[parent] = [chain]
            else:
                parent_map[parent].append(chain)

    for parent, chains in parent_map.items():
        if len(chains) > 1:
            if chains[0].start_node != parent.left:
                tmp = chains[0]
                chains[0] = chains[1]
                chains[1] = tmp

    return parent_map

def inter_chain_merge(parent: Node, left_chain: MergeChain, right_chain: MergeChain):
    if parent.left != left_chain.start_node or parent.right != right_chain.start_node:
        raise ValueError('Parent not match')

    feature_id = parent.feature_id

    left_node = left_chain.end_node
    while left_node != parent:
        if left_node.feature_id == feature_id and left_node.right.mode == b'LEAF':
            if left_chain.value == 2 or left_chain.value == int(left_node.right.target_weight):
                break
            return
        left_node = left_node.parent
    if left_node == parent:
        return

    right_node = right_chain.end_node
    while right_node != parent:
        if right_node.feature_id == feature_id and right_node.left.mode == b'LEAF':
            if right_chain.value == 2 or right_chain.value == int(right_node.left.target_weight):
                break
            return
        right_node = right_node.parent
    if right_node == parent:
        return
    
    print('inter-chain merging...')

    # inter-chain merge
    left_length = 0
    node = left_node
    while node != parent:
        left_length += 1
        node = node.parent

    right_length = 0
    node = right_node
    while node != parent:
        right_length += 1
        node = node.parent

    # TODO: here we should use cardinality estimation, current code is a heuristic solution
    # TODO: this may cause new chain, later we should do intra-chain merge and inter-chain merge together
    if left_length <= right_length:  # right -> left
        # parent
        parent.value = right_node.value
        
        # right_node
        if right_node == right_node.parent.left:
            right_node.parent.left = right_node.right
        else:
            right_node.parent.right = right_node.right
        right_node.right.parent = right_node.parent
        
        # left_node
        left_node.right.samples += right_node.left.samples

        # right_chain
        # only one node in the chain
        if right_chain.start_node == right_chain.end_node:
            right_chain.start_node = None
            right_chain.end_node = None
            right_chain.value = None
        else:
            if right_node == right_chain.start_node:
                right_chain.start_node = right_node.right
            if right_node == right_chain.end_node:
                right_chain.end_node = right_node.parent
            right_chain.check_and_update_value()

    else:  # left -> right
        # parent
        parent.value = left_node.value

        # left_node
        if left_node == left_node.parent.left:
            left_node.parent.left = left_node.left
        else:
            left_node.parent.right = left_node.left
        left_node.left.parent = left_node.parent

        # right_node
        right_node.left.samples += left_node.right.samples

        # left_chain
        # only one node in the chain
        if left_chain.start_node == left_chain.end_node:
            left_chain.start_node = None
            left_chain.end_node = None
            left_chain.value = None
        else:
            if left_node == left_chain.start_node:
                left_chain.start_node = left_node.left
            if left_node == left_chain.end_node:
                left_chain.end_node = left_node.parent
            left_chain.check_and_update_value()

    left_chain.update_samples()
    right_chain.update_samples()

start = time.perf_counter()
root = model2tree(model, samples_list, 0, None)

# only for debug
def debug_samples(root: 'Node') :
    if root.mode != b'LEAF':
        if root.samples != debug_samples(root.left) + debug_samples(root.right):
            raise ValueError('Samples not match')
    return root.samples

reduced_cost = 0

merge_chains: List[MergeChain] = []
find_merge_chains(root, merge_chains)
for i, merge_chain in enumerate(merge_chains):
    print(i)
    merge_chain.print()
    reduced_cost += merge_chain.start_node.branch_samples()
    merge_chain.merge()
    reduced_cost -= merge_chain.start_node.branch_samples()
    merge_chain.print()

chain_parent_map = list_to_parent_map(merge_chains)
i = 0
for parent, chains in chain_parent_map.items():
    if len(chains) > 1:
        print(i)
        i += 1
        print('parent:', MergeChain.node_str(parent))
        print('left:')
        chains[0].print()
        print('right:')
        chains[1].print()
        reduced_cost += parent.left.branch_samples() + parent.right.branch_samples()
        inter_chain_merge(parent, chains[0], chains[1])
        reduced_cost -= parent.left.branch_samples() + parent.right.branch_samples()
        print('parent:', MergeChain.node_str(parent))
        print('left:')
        chains[0].print()
        print('right:')
        chains[1].print()

        debug_samples(parent)

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

debug_samples(root)

print(f'Performance: {sum(samples_list) / (sum(samples_list) - reduced_cost)}')
