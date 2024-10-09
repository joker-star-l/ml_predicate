import onnx
import time
import pandas as pd
from typing import List, Tuple
import argparse
from tree import Node, TreeEnsembleRegressor, model2tree

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='Ailerons_d10_l703_n1405_20240915180213')
args = parser.parse_args()

model_name = args.model

alpha = 1

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'

model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

class DPNode:
    def __init__(self, idx, k, node_samples, branch_samples, dp_value, threshold, flag, node):
        self.idx: Tuple[int, int] = idx  # 索引
        self.k: int = k  # 分割点
        self.node_samples: int = node_samples  # 节点样本数
        self.branch_samples: int = branch_samples  # 节点及其分支样本总数
        self.dp_value: float = dp_value  # DP值
        self.threshold: float = threshold  # 阈值
        self.flag: int = flag  # 标志 0: False; 1: True; 2: Both;

        self.node: 'Node | None' = node  # 新的真实节点
    
    @staticmethod
    def get_flag_from_node(node: 'Node') -> int:
        if node.mode != b'LEAF':
            return 2
        return int(node.target_weight)
    
    @staticmethod
    def show(dp_arr: List[List['DPNode | None']]):
        DPNode.show_internal(dp_arr, 0, len(dp_arr) - 1)
        print()

    @staticmethod
    def show_internal(dp_arr: List[List['DPNode | None']], i: int, j: int):
        if i == j:
            print(f'A{dp_arr[i][i].flag}', end='')
        else:
            print('(', end='')
            DPNode.show_internal(dp_arr, i, dp_arr[i][j].k)
            DPNode.show_internal(dp_arr, dp_arr[i][j].k + 1, j)
            print(')', end='')

class SubTreeForDP:
    def __init__(self, root, count):
        self.root: 'Node' = root
        self.feature_id: int = self.root.feature_id
        self.parent: 'Node' | None = self.root.parent
        self.count: int = count

        self.ordered_internal_nodes: List['Node'] = []
        self.ordered_leaf_nodes: List['Node'] = []
        SubTreeForDP.init_nodes(root, self.feature_id, self.ordered_internal_nodes, self.ordered_leaf_nodes)

        # only for debug
        if len(self.ordered_internal_nodes) != count or len(self.ordered_leaf_nodes) != count + 1:
            raise ValueError('Count not match')

        self.dp_arr: List[List['DPNode | None']] = None

    @staticmethod
    def init_nodes(node: 'Node', feature_id: int, ordered_internal_nodes: 'List[Node]', ordered_leaf_nodes: 'List[Node]'):
        # 左根右遍历，保证阈值从小到大排列
        if node.mode != b'LEAF' and node.feature_id == feature_id:
            SubTreeForDP.init_nodes(node.left, feature_id, ordered_internal_nodes, ordered_leaf_nodes)
            ordered_internal_nodes.append(node)
            SubTreeForDP.init_nodes(node.right, feature_id, ordered_internal_nodes, ordered_leaf_nodes)
        else:
            ordered_leaf_nodes.append(node)

    def dp(self):
        n = self.count + 1
        dp_arr: List[List['DPNode | None']] = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            node = self.ordered_leaf_nodes[i]
            dp_arr[i][i] = DPNode(
                (i, i),
                i,
                node.samples,
                node.samples,
                float(node.samples),
                node.value,
                DPNode.get_flag_from_node(node),
                node
            )
        for l in range(2, n + 1):  # l: length
            for i in range(n - l + 1):
                j = i + l - 1
                dp_arr[i][j] = None
                for k in range(i, j):
                    left = dp_arr[i][k]
                    right = dp_arr[k + 1][j]
                    branch_samples = left.branch_samples + right.branch_samples
                    dp_value = left.dp_value + right.dp_value
                    
                    flag = 2
                    if left.flag == right.flag and left.flag != 2:
                        flag = left.flag
                    else:
                        branch_samples += left.node_samples + right.node_samples
                        dp_value += alpha * left.node_samples + right.node_samples
                    
                    if dp_arr[i][j] is None or dp_value < dp_arr[i][j].dp_value:
                        dp_arr[i][j] = DPNode(
                            (i, j),
                            k,
                            left.node_samples + right.node_samples,
                            branch_samples,
                            dp_value,
                            self.ordered_internal_nodes[k].value,
                            flag,
                            None
                        )
        self.dp_arr = dp_arr

    def change_nodes(self):
        new_root = self.change_nodes_internal(0, self.count).node
        new_root.parent = self.parent
        if self.parent is not None:
            if self.parent.left == self.root:
                self.parent.left = new_root
            else:
                self.parent.right = new_root
        self.root = new_root

    def change_nodes_internal(self, i: int, j: int) -> 'DPNode':
        dp_node = self.dp_arr[i][j]

        # 叶子节点
        if i == j:
            return dp_node

        # dp之前是非叶子节点，需要判断dp之后是否是非叶子节点
        node_mode = b'BRANCH_LEQ' if dp_node.flag == 2 else b'LEAF'

        if node_mode != b'LEAF':
            dp_left = self.change_nodes_internal(i, dp_node.k)
            dp_right = self.change_nodes_internal(dp_node.k + 1, j)

        old_node = self.ordered_internal_nodes[dp_node.k]
        
        # only for debug
        if old_node.value != dp_node.threshold:
            raise ValueError('threshold not match')

        new_node = Node(
            old_node.id,
            old_node.feature_id,
            node_mode,
            0 if node_mode == b'LEAF' else old_node.value,
            None,
            float(dp_node.flag) if node_mode == b'LEAF' else None,
            dp_node.node_samples,
        )

        if node_mode != b'LEAF':
            # only for debug
            if dp_left is None or dp_right is None:
                raise ValueError('dp_left or dp_right is None')

            new_node.left = dp_left.node
            dp_left.node.parent = new_node

            new_node.right = dp_right.node
            dp_right.node.parent = new_node

        dp_node.node = new_node

        return dp_node

def get_same_feature_subtrees(node: 'Node', sub_roots: List[Tuple['Node', int]]) -> int:
    if node.mode == b'LEAF':
        return 0

    # 非叶子节点
    count = 1 + get_same_feature_subtrees(node.left, sub_roots) + get_same_feature_subtrees(node.right, sub_roots)
    if node.parent is not None and node.parent.feature_id == node.feature_id:
        return count
    else:
        if count > 1:
            sub_roots.append((node, count))

            # only for debug: get depth
            depth = 0
            p_node = node
            while p_node.parent is not None:
                depth += 1
                p_node = p_node.parent
            print(f'feature: {node.feature_id}, subtree_root_depth: {depth}, count: {count}')

        return 0

start = time.perf_counter()
root = model2tree(model, samples_list, 0, None)
before_cost = root.cost(alpha)

sub_roots: List[Tuple['Node', int]] = []
reduced_cost = 0
get_same_feature_subtrees(root, sub_roots)

for sub_root, count in sub_roots:
    subtree = SubTreeForDP(sub_root, count)
    subtree.dp()
    subtree.change_nodes()
    DPNode.show(subtree.dp_arr)

    # change root
    if sub_root.parent is None:
        root = subtree.root

    # only for debug
    if subtree.root.same_feature_branch_samples() != subtree.dp_arr[0][count].branch_samples:
        raise ValueError('Branch samples not match')

    print(f'cost: {sub_root.same_feature_branch_samples()}, {subtree.root.same_feature_branch_samples()}, {subtree.dp_arr[0][count].branch_samples}')
    reduced_cost += sub_root.same_feature_branch_samples() - subtree.dp_arr[0][count].branch_samples

after_cost = root.cost(alpha)
regressor = TreeEnsembleRegressor.from_tree(root)
output_model = regressor.to_model(model)
onnx.save_model(output_model, model_path.replace('_out.onnx', '_out2.onnx'))
end = time.perf_counter()

print(f'Elapsed time: {end - start:.6f}s')

print(f'Branch samples: {root.branch_samples()}', sum(samples_list))

print(f'Reduced cost: {before_cost - after_cost}')

# only for debug TODO
if root.branch_samples() + reduced_cost != sum(samples_list):
    raise ValueError('Branch samples not match')

print(f'Performance: {before_cost / after_cost}')
