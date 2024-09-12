import onnx
from utils import get_attribute

model_name = 'house_16H_d10_l405_n809_20240912085537_out'
model_path = f'model_output/{model_name}'

model = onnx.load(model_path + '.onnx')
# nodes_truenodeids: 左侧分支
nodes_truenodeids = get_attribute(model, 'nodes_truenodeids').ints
# nodes_falsenodeids: 右侧分支
nodes_falsenodeids = get_attribute(model, 'nodes_falsenodeids').ints
# nodes_featureids: 特征id
nodes_featureids = get_attribute(model, 'nodes_featureids').ints
# nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
node_modes = get_attribute(model, 'nodes_modes').strings

# target_nodeids: 叶子节点的id
target_nodeids = get_attribute(model, 'target_nodeids').ints
target_nodeid_map = {node_id: i for i, node_id in enumerate(target_nodeids)}
# target_weights: 叶子节点的权重，即预测值
target_weights = get_attribute(model, 'target_weights').floats

can_rotate = [0]
can_merge = [0]
not_leaf = [0]

def count_can_rotate(node_id: int, feature_id: int, mode: str):    
    global can_rotate
    global can_merge
    global not_leaf

    if mode == b'LEAF':
        return
    
    not_leaf[0] += 1
    
    left_node_id = nodes_truenodeids[node_id]
    left_mode = node_modes[left_node_id]
    
    right_node_id = nodes_falsenodeids[node_id]
    right_mode = node_modes[right_node_id]

    if left_mode != b'LEAF':
        left_feature_id = nodes_featureids[left_node_id]
        if left_feature_id == feature_id:
            can_rotate[0] += 1

            if right_mode == b'LEAF':
                left_right_id = nodes_falsenodeids[left_node_id]
                left_right_mode = node_modes[left_right_id]
                if left_right_mode == b'LEAF' and target_weights[target_nodeid_map[right_node_id]] == target_weights[target_nodeid_map[left_right_id]]:
                    can_merge[0] += 1

        count_can_rotate(left_node_id, left_feature_id, left_mode)
    
    if right_mode != b'LEAF':
        right_feature_id = nodes_featureids[right_node_id]
        if right_feature_id == feature_id:
            can_rotate[0] += 1

            if left_mode == b'LEAF':
                right_left_id = nodes_truenodeids[right_node_id]
                right_left_mode = node_modes[right_left_id]
                if right_left_mode == b'LEAF' and target_weights[target_nodeid_map[left_node_id]] == target_weights[target_nodeid_map[right_left_id]]:
                    can_merge[0] += 1

        count_can_rotate(right_node_id, right_feature_id, right_mode)        


count_can_rotate(0, nodes_featureids[0], node_modes[0])
print(f"can_rotate: {can_rotate}")
print(f"can_merge: {can_merge}")
print(f"not_leaf: {not_leaf}")
print(f"nodes: {len(nodes_featureids)}")
