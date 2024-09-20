#  该代码废弃，旋转难以解决全部的情况，需要使用动态规划

import onnx
import time
from onnx import helper
import pandas as pd
from utils import get_attribute
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l448_n895_20240919101404')
args = parser.parse_args()

model_name = args.model

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'


model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

class Node:
    def __init__(
            self,
            id,  # 节点id
            feature_id,  # 特征id
            mode,  # 节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
            value,  # 阈值，叶子节点的值为0
            target_id,  # 叶子节点的taget id
            target_weight,  # 叶子节点的权重，即预测值
            samples  # 节点的样本数
            ):
        self.id: int = id
        self.feature_id: int = feature_id
        self.mode: bytes = mode
        self.value: float = value
        self.target_id: int | None = target_id
        self.target_weight: float | None = target_weight
        self.samples: int = samples
        
        self.parent: 'Node' | None = None
        self.left: 'Node' | None = None
        self.right: 'Node' | None  = None

    def branch_samples(self) -> int:
        samples = self.samples
        
        if self.left is not None:
            samples += self.left.branch_samples()
        if self.right is not None:
            samples += self.right.branch_samples()
        
        return samples

class TreeEnsembleRegressor:
    def __init__(self):
        self.n_targets: int = 1
        self.nodes_falsenodeids: list[int] = []
        self.nodes_featureids: list[int] = []
        self.nodes_hitrates: list[float] = []
        self.nodes_missing_value_tracks_true: list[int] = []
        self.nodes_modes: list[bytes] = []
        self.nodes_nodeids: list[int] = []
        self.nodes_treeids: list[int] = []
        self.nodes_truenodeids: list[int] = []
        self.nodes_values: list[float] = []
        self.post_transform: bytes = b'NONE'
        self.target_ids: list[int] = []
        self.target_nodeids: list[int] = []
        self.target_treeids: list[int] = []
        self.target_weights: list[float] = []

    def to_model(self, input_model: onnx.ModelProto) -> onnx.ModelProto:
        # node
        node = helper.make_node(
            op_type='TreeEnsembleRegressor',
            inputs=[input_model.graph.input[0].name],
            outputs=[input_model.graph.output[0].name],
            name='TreeEnsembleRegressor',
            domain='ai.onnx.ml',
            # attributes
            n_targets=self.n_targets,
            nodes_falsenodeids=self.nodes_falsenodeids,
            nodes_featureids=self.nodes_featureids,
            nodes_hitrates=self.nodes_hitrates,
            nodes_missing_value_tracks_true=self.nodes_missing_value_tracks_true,
            nodes_modes=self.nodes_modes,
            nodes_nodeids=self.nodes_nodeids,
            nodes_treeids=self.nodes_treeids,
            nodes_truenodeids=self.nodes_truenodeids,
            nodes_values=self.nodes_values,
            post_transform=self.post_transform,
            target_ids=self.target_ids,
            target_nodeids=self.target_nodeids,
            target_treeids=self.target_treeids,
            target_weights=self.target_weights
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

        return output_model

    @staticmethod
    def from_tree(root: 'Node') -> 'TreeEnsembleRegressor':
        regressor = TreeEnsembleRegressor()
        TreeEnsembleRegressor.from_tree_internal(regressor, root)
        
        id_map = {old_id: i for i, old_id in enumerate(regressor.nodes_nodeids)}
        print(id_map)
        is_leaf = [mode == b'LEAF' for mode in regressor.nodes_modes]
        regressor.nodes_falsenodeids = [(0 if is_leaf[i] else id_map[id]) for i, id in enumerate(regressor.nodes_falsenodeids)]
        regressor.nodes_truenodeids = [(0 if is_leaf[i] else id_map[id]) for i, id in enumerate(regressor.nodes_truenodeids)]
        regressor.nodes_nodeids = [id_map[id] for id in regressor.nodes_nodeids]
        regressor.target_nodeids = [id_map[id] for id in regressor.target_nodeids]
        
        return regressor

    @staticmethod
    def from_tree_internal(regressor: 'TreeEnsembleRegressor', node: 'Node'):
        is_leaf = node.mode == b'LEAF'

        regressor.nodes_falsenodeids.append(node.right.id if not is_leaf else 0)
        regressor.nodes_featureids.append(node.feature_id)
        regressor.nodes_hitrates.append(1.0)
        regressor.nodes_missing_value_tracks_true.append(0)
        regressor.nodes_modes.append(node.mode)
        regressor.nodes_nodeids.append(node.id)
        regressor.nodes_treeids.append(0)
        regressor.nodes_truenodeids.append(node.left.id if not is_leaf else 0)
        regressor.nodes_values.append(node.value)
        
        if is_leaf:
            regressor.target_ids.append(0)
            regressor.target_nodeids.append(node.id)
            regressor.target_treeids.append(0)
            regressor.target_weights.append(node.target_weight)

        if not is_leaf:
            TreeEnsembleRegressor.from_tree_internal(regressor, node.left)
            TreeEnsembleRegressor.from_tree_internal(regressor, node.right)

def model2tree(input_model, samples_list, node_id, parent: 'Node | None') -> 'Node':
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

    # node_id -> target_id
    input_target_nodeid_map = {node_id: i for i, node_id in enumerate(input_target_nodeids)}

    id = node_id
    feature_id = input_nodes_featureids[id]
    mode = input_node_modes[id]
    value = input_nodes_values[id]
    target_id = input_target_nodeid_map.get(id, None)
    target_weight = input_target_weights[target_id] if target_id is not None else None
    samples = samples_list[id]
    node = Node(
        id=id,
        feature_id=feature_id,
        mode=mode,
        value=value,
        target_id=target_id,
        target_weight=target_weight,
        samples=samples
    )
    node.parent = parent
    
    if mode != b'LEAF':
        left_node_id = input_nodes_truenodeids[id]
        left_node = model2tree(input_model, samples_list, left_node_id, node)
        node.left = left_node

        right_node_id = input_nodes_falsenodeids[id]
        right_node = model2tree(input_model, samples_list, right_node_id, node)
        node.right = right_node

    return node

reduced_cost = 0

def rotate(node: 'Node', root: list['Node']):
    global reduced_cost

    print(node.id)
    
    if node.mode != b'LEAF':
        rotate(node.left, root)
        rotate(node.right, root)

        print(f'doing parent {node.id}, left {node.left.id}, right {node.right.id}')

        left_is_leaf = node.left.mode == b'LEAF'
        feature_same_with_left = (not left_is_leaf) and node.left.feature_id == node.feature_id
        right_is_leaf = node.right.mode == b'LEAF'
        feature_same_with_right = (not right_is_leaf) and node.right.feature_id == node.feature_id

        if feature_same_with_left and (not feature_same_with_right):  # 与左子节点特征相同
            print('feature_same_with_left')
            
            left_right_is_leaf = node.left.right.mode == b'LEAF'
            can_merge = right_is_leaf and left_right_is_leaf and node.left.right.target_weight == node.right.target_weight
            
            if can_merge:  # 可以合并，直接旋转并合并
                print('can_merge:', node.left.samples)
                reduced_cost += node.left.samples
                
                left = node.left
                
                left.parent = node.parent
                if node.parent is not None:
                    if left.parent.left == node:
                        left.parent.left = left
                    else:
                        left.parent.right = left
                else:
                    root[0] = left  # 旋转后的根节点
                node.parent = None
                node.left = None

                left.samples = node.samples
                left.right.samples += node.right.samples

            else:  # 不能合并，需要计算代价判断是需要旋转
                print('cannot_merge')

                if node.left.left.samples > node.right.samples:
                    print('need_rotate:', node.left.left.samples, node.right.samples)
                    reduced_cost += (node.left.left.samples - node.right.samples)

                    left = node.left
                    
                    left.parent = node.parent
                    if node.parent is not None:
                        if left.parent.left == node:
                            left.parent.left = left
                        else:
                            left.parent.right = left
                    else:
                        root[0] = left  # 旋转后的根节点
                    node.parent = left
                    node.left = left.right
                    left.right.parent = node
                    left.right = node

                    left.samples = node.samples
                    node.samples = node.left.samples + node.right.samples
                
                else:
                    print('donot_need_rotate:', node.left.left.samples, node.right.samples)

        elif (not feature_same_with_left) and feature_same_with_right:  # 与右子节点特征相同
            print('feature_same_with_right')

            right_left_is_leaf = node.right.left.mode == b'LEAF'
            can_merge = left_is_leaf and right_left_is_leaf and node.right.left.target_weight == node.left.target_weight

            if can_merge:  # 可以合并，直接旋转并合并
                print('can_merge:', node.right.samples)
                reduced_cost += node.right.samples

                right = node.right

                right.parent = node.parent
                if node.parent is not None:
                    if right.parent.left == node:
                        right.parent.left = right
                    else:
                        right.parent.right = right
                else:
                    root[0] = right  # 旋转后的根节点
                node.parent = None
                node.right = None

                right.samples = node.samples
                right.left.samples += node.left.samples

            else:  # 不能合并，需要计算代价判断是需要旋转
                print('cannot_merge')

                if node.right.right.samples > node.left.samples:
                    print('need_rotate:', node.right.right.samples, node.left.samples)
                    reduced_cost += (node.right.right.samples - node.left.samples)

                    right = node.right

                    right.parent = node.parent
                    if node.parent is not None:
                        if right.parent.left == node:
                            right.parent.left = right
                        else:
                            right.parent.right = right
                    else:
                        root[0] = right  # 旋转后的根节点
                    node.parent = right
                    node.right = right.left
                    right.left.parent = node
                    right.left = node

                    right.samples = node.samples
                    node.samples = node.left.samples + node.right.samples

                else:
                    print('donot_need_rotate:', node.right.right.samples, node.left.samples)

        elif feature_same_with_left and feature_same_with_right:  # 都相同
            print('feature_same_with_left_and_right')

            left_right_is_leaf = node.left.right.mode == b'LEAF'
            right_left_is_leaf = node.right.left.mode == b'LEAF'
            can_merge = left_right_is_leaf and right_left_is_leaf and node.left.right.target_weight == node.right.left.target_weight

            if can_merge:  # 可以合并，直接旋转并合并
                print('can_merge')

                if node.left.left.samples <= node.right.right.samples: # 左结合
                    print('left_merge:', node.right.right.samples)
                    reduced_cost += node.right.right.samples

                else:  # 右结合
                    print('right_merge:', node.left.left.samples)
                    reduced_cost += node.left.left.samples

            else:  # 不能合并，需要计算代价判断是需要旋转
                print('cannot_merge')

                cost_origin = node.left.samples + node.right.samples                                              # ((ab)(cd))
                cost_case_a = node.left.samples * 2 + node.right.left.samples                                     # (((ab)c)d)
                cost_case_b = (node.left.right.samples + node.right.left.samples) * 2 + node.left.left.samples    # ((a(bc))d)
                cost_case_c = (node.left.right.samples + node.right.left.samples) * 2 + node.right.right.samples  # (a((bc)d))
                cost_case_d = node.right.samples * 2 + node.left.right.samples                                    # (a(b(cd)))

                cost_min = min(cost_origin, cost_case_a, cost_case_b, cost_case_c, cost_case_d)
                if cost_min == cost_origin:
                    print('donot_need_rotate ((ab)(cd)):', cost_origin, cost_min)
                
                elif cost_min == cost_case_a:
                    print('need_rotate (((ab)c)d):', cost_origin, cost_min)
                    reduced_cost += (cost_origin - cost_min)
                
                elif cost_min == cost_case_b:
                    print('need_rotate ((a(bc))d):', cost_origin, cost_min)
                    reduced_cost += (cost_origin - cost_min)
                
                elif cost_min == cost_case_c:
                    print('need_rotate (a((bc)d)):', cost_origin, cost_min)
                    reduced_cost += (cost_origin - cost_min)
                
                elif cost_min == cost_case_d:
                    print('need_rotate (a(b(cd))):', cost_origin, cost_min)
                    reduced_cost += (cost_origin - cost_min)


start = time.perf_counter()
root = model2tree(model, samples_list, 0, None)
new_root = [root]  # keep the root node
rotate(root, new_root)
root = new_root[0]
regressor = TreeEnsembleRegressor.from_tree(root)
output_model = regressor.to_model(model)
onnx.save_model(output_model, model_path.replace('_out.onnx', '_out2.onnx'))
end = time.perf_counter()

print(f'Elapsed time: {end - start:.6f}s')

print(f'Branch samples: {root.branch_samples()}', sum(samples_list))

print(f'Reduced cost: {reduced_cost}')

print(f'Performance: {sum(samples_list) / (sum(samples_list) - reduced_cost)}')
