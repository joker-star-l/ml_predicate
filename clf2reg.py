import onnx
import joblib
from onnx import helper
import onnx.checker
import argparse
import pandas as pd
import numpy as np

from utils import get_attribute

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='bank-marketing_d3_l8_n15_20241104075020')
args = parser.parse_args()

model_path = f'model/{args.model}'
out_path = f'model_output/{args.model}_reg'

def clf2reg(input_model: onnx.ModelProto) -> onnx.ModelProto:
    # input model attributes
    # # class_ids: 叶子节点权重对应的类别id
    input_class_ids = get_attribute(input_model, 'class_ids').ints
    # # class_nodeids: 叶子节点权重对应的节点id
    input_class_nodeids = get_attribute(input_model, 'class_nodeids').ints
    # # class_treeids: 叶子节点权重对应的树id
    input_class_treeids = get_attribute(input_model, 'class_treeids').ints
    # # class_weights: 叶子节点权重，即预测值
    input_class_weights = get_attribute(input_model, 'class_weights').floats
    # # classlabels_int64s: 类别id
    input_classlabels_int64s = get_attribute(input_model, 'classlabels_int64s').ints
    # # nodes_falsenodeids: 右侧分支
    input_nodes_falsenodeids = get_attribute(input_model, 'nodes_falsenodeids').ints
    # # nodes_featureids: 特征id
    input_nodes_featureids = get_attribute(input_model, 'nodes_featureids').ints
    # # nodes_hitrates
    input_nodes_hitrates = get_attribute(input_model, 'nodes_hitrates').floats
    # # nodes_missing_value_tracks_true
    input_nodes_missing_value_tracks_true = get_attribute(input_model, 'nodes_missing_value_tracks_true').ints
    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    input_nodes_modes = get_attribute(input_model, 'nodes_modes').strings
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

    # output model attributes
    # # n_targets
    n_targets = 1

    # # nodes_falsenodeids: 右侧分支
    nodes_falsenodeids = input_nodes_falsenodeids

    # # nodes_featureids: 特征id
    nodes_featureids = input_nodes_featureids

    # # nodes_hitrates
    nodes_hitrates = input_nodes_hitrates

    # # nodes_missing_value_tracks_true
    nodes_missing_value_tracks_true = input_nodes_missing_value_tracks_true

    # # nodes_modes：节点类型，LEAF表示叶子节点，BRANCH_LEQ表示非叶子节点
    nodes_modes = input_nodes_modes

    # # nodes_nodeids
    nodes_nodeids = input_nodes_nodeids

    # # nodes_treeids
    nodes_treeids = input_nodes_treeids

    # # nodes_truenodeids: 左侧分支
    nodes_truenodeids = input_nodes_truenodeids

    # # nodes_values: 阈值，叶子节点的值为0
    nodes_values = input_nodes_values

    # # post_transform
    post_transform = input_post_transform

    stride = len(input_classlabels_int64s)
    if stride == 2:
        stride = 1

    n_leaf = len(input_class_weights) // stride

    # # target_ids
    target_ids = []
    for i in range(n_leaf):
        target_ids.append(input_class_ids[i * stride])

    # # target_nodeids: 叶子节点的id
    target_nodeids = []
    for i in range(n_leaf):
        target_nodeids.append(input_class_nodeids[i * stride])

    # # target_treeids
    target_treeids = []
    for i in range(n_leaf):
        target_treeids.append(input_class_treeids[i * stride])
    
    # # target_weights: 叶子节点的权重，即预测值
    target_weights = []
    if stride == 1:
        # binary mode: only store positive class weight
        target_weights = [1.0 if w > 0.5 else 0.0 for w in input_class_weights]
    else:
        for i in range(n_leaf):
            targets = input_class_weights[i * stride: (i + 1) * stride]
            target_weights.append(float(np.argmax(targets)))
    
    # node
    node = helper.make_node(
        op_type='TreeEnsembleRegressor',
        inputs=[input_model.graph.input[0].name],
        outputs=[input_model.graph.output[0].name],
        name='TreeEnsembleRegressor',
        domain='ai.onnx.ml',
        # attributes
        n_targets=n_targets,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        post_transform=post_transform,
        target_ids=target_ids,
        target_nodeids=target_nodeids,
        target_treeids=target_treeids,
        target_weights=target_weights
    )

    # graph
    output = helper.make_tensor_value_info(
        name=input_model.graph.output[0].name,
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, 1],
    )
    graph = helper.make_graph(
        nodes=[node],
        name=input_model.graph.name,
        initializer=[],
        inputs=input_model.graph.input,
        outputs=[output],
    )

    # model
    output_model = helper.make_model(
        graph=graph,
        opset_imports=input_model.opset_import,
    )
    output_model.ir_version = input_model.ir_version

    onnx.checker.check_model(output_model)

    return output_model

model = onnx.load(model_path + '.onnx')
output_model = clf2reg(model)
onnx.save_model(output_model, out_path + '.onnx')
