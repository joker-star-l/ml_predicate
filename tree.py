import onnx
from onnx import helper
from utils import get_attribute

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
