import onnx
from onnx import helper
# from utils import get_attribute
from typing import List, Tuple

def get_attribute(onnx_model, attr_name):
    attributes = onnx_model.graph.node[0].attribute
    for attr in attributes:
        if attr.name == attr_name:
            return attr
            
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


def model2tree(input_model, samples_list: 'List[int] | None', node_id, parent: 'Node | None', 
               tree_interval: 'Tuple[int, int] | None' = None, target_tree_interval: 'Tuple[int, int] | None' = None) -> 'Node':
    if tree_interval is None:
        tree_interval = (0, len(get_attribute(input_model, 'nodes_treeids').ints))
    tree_start, tree_end = tree_interval

    if target_tree_interval is None:
        target_tree_interval = (0, len(get_attribute(input_model, 'target_treeids').ints))
    target_tree_start, target_tree_end = target_tree_interval

    # input model attributes
    # # n_targets
    input_n_targets = get_attribute(input_model, 'n_targets').i
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints[tree_start:tree_end]
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints[tree_start:tree_end]
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, 'nodes_hitrates').floats[tree_start:tree_end]
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(input_model, 'nodes_missing_value_tracks_true').ints[tree_start:tree_end]
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_node_modes = get_attribute(input_model, 'nodes_modes').strings[tree_start:tree_end]
    # # nodes_nodeids
    input_nodes_nodeids = get_attribute(input_model, 'nodes_nodeids').ints[tree_start:tree_end]
    # # nodes_treeids
    input_nodes_treeids = get_attribute(input_model, 'nodes_treeids').ints[tree_start:tree_end]
    # # nodes_truenodeids: 左侧分支
    input_nodes_truenodeids = get_attribute(input_model, 'nodes_truenodeids').ints[tree_start:tree_end]
    # # nodes_values: 阈值，叶子节点的值为0
    input_nodes_values = get_attribute(input_model, 'nodes_values').floats[tree_start:tree_end]
    # # post_transform
    input_post_transform = get_attribute(input_model, 'post_transform').s
    # # target_ids
    input_target_ids = get_attribute(input_model, 'target_ids').ints[target_tree_start:target_tree_end]
    # # target_nodeids: 叶子节点的id
    input_target_nodeids = get_attribute(input_model, 'target_nodeids').ints[target_tree_start:target_tree_end]
    # # target_treeids
    input_target_treeids = get_attribute(input_model, 'target_treeids').ints[target_tree_start:target_tree_end]
    # # target_weights: 叶子节点的权重，即预测值
    input_target_weights = get_attribute(input_model, 'target_weights').floats[target_tree_start:target_tree_end]

    # node_id -> target_id
    input_target_nodeid_map = {node_id: i for i, node_id in enumerate(input_target_nodeids)}

    id = node_id
    feature_id = input_nodes_featureids[id]
    mode = input_node_modes[id]
    value = input_nodes_values[id]
    target_id = input_target_nodeid_map.get(id, None)
    target_weight = input_target_weights[target_id] if target_id is not None else None
    samples = int(input_nodes_hitrates[id])
    
    
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
        left_node = model2tree(input_model, samples_list, left_node_id, node, tree_interval, target_tree_interval)
        node.left = left_node

        right_node_id = input_nodes_falsenodeids[id]
        right_node = model2tree(input_model, samples_list, right_node_id, node, tree_interval, target_tree_interval)
        node.right = right_node
        
    return node