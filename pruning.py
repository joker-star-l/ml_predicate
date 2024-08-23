import onnx
import joblib
from onnx import helper
import onnx.checker


# node attributes
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

model_path = 'model/house_16H_d5_l25_n49_20240821155130'
func = lambda x: x > 10


def get_attribute(onnx_model, attr_name):
    attributes = onnx_model.graph.node[0].attribute
    for attr in attributes:
        if attr.name == attr_name:
            return attr


def pruning(node_id, depth, result_nodes, onnx_model, joblib_model, f) -> int:  # 0: leaf_false, 1: leaf_true, 2: inner
    left_nodes = get_attribute(onnx_model, 'nodes_truenodeids').ints
    right_nodes = get_attribute(onnx_model, 'nodes_falsenodeids').ints
    node_types = get_attribute(onnx_model, 'nodes_modes').strings
    node_thresholds = get_attribute(onnx_model, 'nodes_values').floats

    target_nodeids = get_attribute(onnx_model, 'target_nodeids').ints
    target_weights = get_attribute(onnx_model, 'target_weights').floats

    result_nodes[node_id] = node_types[node_id].decode('utf-8')
    is_leaf = node_types[node_id] == b'LEAF'

    if is_leaf:
        target_id = -1
        for ti, ni in enumerate(target_nodeids):
            if ni == node_id:
                target_id = ti
                break

        result = int(f(target_weights[target_id]))

        # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {result}')

        return result

    if not is_leaf:
        left_node_id = left_nodes[node_id]
        left_result = pruning(left_node_id, depth + 1, result_nodes, onnx_model, joblib_model, f)
        right_node_id = right_nodes[node_id]
        right_result = pruning(right_node_id, depth + 1, result_nodes, onnx_model, joblib_model, f)

        if left_result == 0 and right_result == 0:
            # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {0}')
            result_nodes[node_id] = 'LEAF'
            result_nodes[left_node_id] = 'REMOVED'
            result_nodes[right_node_id] = 'REMOVED'
            return 0

        if left_result == 1 and right_result == 1:
            # print(f'node_id: {node_id}, depth: {depth}, is_leaf: {is_leaf}, result: {1}')
            result_nodes[node_id] = 'LEAF'
            result_nodes[left_node_id] = 'REMOVED'
            result_nodes[right_node_id] = 'REMOVED'
            return 1

        print(f'node_id: {node_id}, depth: {depth}, leaf_depth: {depth + 1}')
        return 2


model = onnx.load(model_path + '.onnx')
# print(model)

result_nodes = [None] * len(get_attribute(model, 'nodes_modes').strings)
pruning(0, 0, result_nodes, model, None, func)
print(result_nodes)
removed_count = result_nodes.count('REMOVED')
print("total nodes:", len(result_nodes), "removed nodes:", removed_count, removed_count / len(result_nodes), f"performance: {len(result_nodes) / (len(result_nodes) - removed_count)}x")

model_joblib = joblib.load(model_path + '.joblib')
node_cost_weights = model_joblib.tree_.n_node_samples
# node_cost_weights = node_cost_weights / node_cost_weights[0]
print(node_cost_weights)
reduced_cost_weights = sum([node_cost_weights[i] for i, node in enumerate(result_nodes) if node == 'REMOVED'])
print("total cost:", sum(node_cost_weights), "reduced cost:", reduced_cost_weights, f"performance: {sum(node_cost_weights) / (sum(node_cost_weights) - reduced_cost_weights)}x")


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

def reg2clf(input_model, removed_nodes, f):    
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
    leaf_count = removed_nodes.count('LEAF')

    new_ids = []
    id_ = 0
    for node in removed_nodes:
        if node == 'LEAF':
            new_ids.append([id_, 'LEAF'])
            id_ += 1
        elif node == 'BRANCH_LEQ':
            new_ids.append([id_, 'BRANCH_LEQ'])
            id_ += 1
        else:
            new_ids.append([-1, 'REMOVED'])

    # # class_ids: 叶子节点权重对应的类别id
    class_ids = helper.make_attribute(
        key='class_ids',
        value=[0] * leaf_count
    )
    onnx.checker.check_attribute(class_ids)

    # # class_nodeids: 叶子节点权重对应的节点id
    class_nodeids = helper.make_attribute(
        key='class_nodeids',
        value=[new_ids[i][0] for i in input_target_nodeids if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(class_nodeids)

    # # class_treeids: 叶子节点权重对应的树id
    class_treeids = helper.make_attribute(
        key='class_treeids',
        value=[0] * leaf_count
    )
    onnx.checker.check_attribute(class_treeids)

    # # class_weights: 叶子节点权重，即预测值
    class_weights = helper.make_attribute(
        key='class_weights',
        value=[int(f(weight)) for i, weight in enumerate(input_target_weights) if new_ids[input_target_nodeids[i]][0] != -1]
    )
    onnx.checker.check_attribute(class_weights)

    # # classlabels_int64s: 类别id
    classlabels_int64s = helper.make_attribute(
        key='classlabels_int64s',
        value=[0, 1]
    )
    onnx.checker.check_attribute(classlabels_int64s)

    # # nodes_falsenodeids: 右侧分支
    nodes_falsenodeids = helper.make_attribute(
        key='nodes_falsenodeids',
        value=[new_ids[ii][0] for i , ii in enumerate(input_nodes_falsenodeids) if new_ids[i][1] != 'REMOVED']
    )
    onnx.checker.check_attribute(nodes_falsenodeids)

    # # nodes_featureids: 特征id
    nodes_featureids = helper.make_attribute(
        key='nodes_featureids',
        value=[ii for i, ii in enumerate(input_nodes_featureids) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_featureids)

    # # nodes_hitrates
    nodes_hitrates = helper.make_attribute(
        key='nodes_hitrates',
        value=[ii for i, ii in enumerate(input_nodes_hitrates) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_hitrates)

    # # nodes_missing_value_tracks_true
    nodes_missing_value_tracks_true = helper.make_attribute(
        key='nodes_missing_value_tracks_true',
        value=[ii for i, ii in enumerate(input_nodes_missing_value_tracks_true) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_missing_value_tracks_true)

    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    nodes_modes = helper.make_attribute(
        key='nodes_modes',
        value=[ii for i, ii in enumerate(input_node_modes) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_modes)

    # # nodes_nodeids
    nodes_nodeids = helper.make_attribute(
        key='nodes_nodeids',
        value=[ii for i, ii in enumerate(input_nodes_nodeids) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_nodeids)

    # # nodes_treeids
    nodes_treeids = helper.make_attribute(
        key='nodes_treeids',
        value=[ii for i, ii in enumerate(input_nodes_treeids) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_treeids)

    # # nodes_truenodeids: 左侧分支
    nodes_truenodeids = helper.make_attribute(
        key='nodes_truenodeids',
        value=[new_ids[ii][0] for i , ii in enumerate(input_nodes_truenodeids) if new_ids[i][1] != 'REMOVED']
    )
    onnx.checker.check_attribute(nodes_truenodeids)

    # # nodes_values: 阈值，叶子节点的值为0
    nodes_values = helper.make_attribute(
        key='nodes_values',
        value=[ii for i, ii in enumerate(input_nodes_values) if new_ids[i][0] != -1]
    )
    onnx.checker.check_attribute(nodes_values)

    # # post_transform
    post_transform = helper.make_attribute(
        key='post_transform',
        value=input_post_transform
    )
    onnx.checker.check_attribute(post_transform)

    # node
    node = helper.make_node(
        op_type='TreeEnsembleClassifier',
        inputs=input_model.graph.input[0].name,
        outputs=['label', 'probabilities'],
        name='TreeEnsembleClassifier',
        domain='ai.onnx.ml',
        # attributes
        class_ids=class_ids,
        class_nodeids=class_nodeids,
        class_treeids=class_treeids,
        class_weights=class_weights,
        classlabels_int64s=classlabels_int64s,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        post_transform=post_transform
    )

    onnx.checker.check_node(node)

    # graph
    label = helper.make_tensor_value_info(
        name='label',
        elem_type=onnx.TensorProto.INT64,
        shape=[None]
    )

    probabilities = helper.make_tensor_value_info(
        name='probabilities',
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, input_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=input_model.graph.name,
        initializer=[],
        inputs=model.graph.input,
        outputs=[label, probabilities],
    )

    onnx.checker.check_graph(graph)
    
    # model
    output_model = helper.make_model(
        graph=graph,
        opset_imports=input_model.opset_import,
    )
    output_model.ir_version = input_model.ir_version

    onnx.checker.check_model(output_model)

    return output_model

output_model = reg2clf(model, result_nodes, func)
print(output_model)