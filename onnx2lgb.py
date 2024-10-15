import onnx
import argparse
import time
import pandas as pd
from utils import get_attribute
from typing import List, Dict
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l859_n1717_20241015054511')
parser.add_argument('--pruned', type=int, default=1)
parser.add_argument('--genc', action='store_true')
args = parser.parse_args()

model_name = args.model
if args.pruned == 0:
    suffix = ''
elif args.pruned == 1:
    suffix = '_out'
elif args.pruned == 2:
    suffix = '_out2'
else:
    raise ValueError('Invalid pruned')

if args.pruned == 0:
    prefix = 'model/'
else:
    prefix = 'model_output/'

onnx_path = f'{prefix}{model_name}{suffix}.onnx'
lgb_path = f'{prefix}{model_name}{suffix}.txt'
featues_path = f'data/features_{model_name[:model_name.find("_d")]}.csv'

def onnx2lgb(input_model: onnx.ModelProto, n_features: int) -> str:
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

    # example in ./lgb_model.txts
    s = ''
    
    s += 'tree\n'
    s += 'version=v4\n'
    s += 'num_class=1\n'
    s += 'num_tree_per_iteration=1\n'
    s += f'max_feature_idx={n_features - 1}\n'
    s += 'objective=regression\n'
    s += f'feature_infos={" ".join(["[:]"] * n_features)}\n'
    s += 'tree_sizes=0\n'
    s += '\n'
   
    s += 'Tree=0\n'
    s += f'num_leaves={len(input_target_nodeids)}\n'
    s += 'num_cat=0\n'

    split_feature_list: List[int] = []
    threshold_list: List[float] = []
    decision_type_list: List[int] = []
    left_child_list: List[int] = []
    right_child_list: List[int] = []
    leaf_value_list: List[float] = []
    leaf_count_list: List[int] = []
    internal_count_list: List[int] = []

    inner_id = 0
    leaf_id = -1
    id_map: Dict[int, int] = {}
    for i, id_ in enumerate(input_nodes_nodeids):
         # only for debug
        if i != id_:
            raise ValueError(f'Invalid node id {i} != {id_}')

        if input_node_modes[i] == b'BRANCH_LEQ':
            id_map[id_] = inner_id
            inner_id += 1
        elif input_node_modes[i] == b'LEAF':
            id_map[id_] = leaf_id
            leaf_id -= 1
        else:
            raise ValueError('Invalid node mode')
        
        # only for debug
        if inner_id + abs(leaf_id) - 1 != id_ + 1:
            raise ValueError(f'Invalid inner_id {inner_id} or leaf_id {leaf_id}')

    # only for debug
    if inner_id + 1 != abs(leaf_id) - 1:
        raise ValueError(f'Invalid inner_id {inner_id} or leaf_id {leaf_id}')

    for i, _ in enumerate(input_nodes_nodeids):
        if input_node_modes[i] == b'BRANCH_LEQ':
            split_feature_list.append(input_nodes_featureids[i])
            threshold_list.append(input_nodes_values[i])
            decision_type_list.append(2)
            left_child_list.append(id_map[input_nodes_truenodeids[i]])
            right_child_list.append(id_map[input_nodes_falsenodeids[i]])
            internal_count_list.append(int(input_nodes_hitrates[i]))
        elif input_node_modes[i] == b'LEAF':
            leaf_value_list.append(input_target_weights[abs(id_map[i]) - 1])
            leaf_count_list.append(int(input_nodes_hitrates[i]))
        else:
            raise ValueError('Invalid node mode')
    
    # only for debug
    if len(leaf_value_list) != len(input_target_nodeids):
        raise ValueError(f'leaf count error: {len(leaf_value_list)}')
    for i in range(len(input_target_nodeids)):
        if input_target_weights[i] != leaf_value_list[i]:
            raise ValueError(f'leaf value error: {input_target_weights[i]} != {leaf_value_list[i]}')

    s += f'split_feature={" ".join([f"{e}" for e in split_feature_list])}\n'
    s += f'threshold={" ".join([f"{e}" for e in threshold_list])}\n'
    s += f'decision_type={" ".join([f"{e}" for e in decision_type_list])}\n'
    s += f'left_child={" ".join([f"{e}" for e in left_child_list])}\n'
    s += f'right_child={" ".join([f"{e}" for e in right_child_list])}\n'
    s += f'leaf_value={" ".join([f"{e}" for e in leaf_value_list])}\n'

    # data_count to generate likely or unlikely c++ code
    s += f'leaf_count={" ".join([f"{e}" for e in leaf_count_list])}\n'
    s += f'internal_count={" ".join([f"{e}" for e in internal_count_list])}\n'

    s += '\n\n'
    s += 'end of trees\n'
    s += '\n'
    s += 'end of parameters\n'
    s += '\n'
    s += 'pandas_categorical:null\n'

    return s

start = time.perf_counter()

onnx_model = onnx.load(onnx_path)
n_features = pd.read_csv(featues_path).shape[0]
lgb_model = onnx2lgb(onnx_model, n_features)
with open(lgb_path, 'w', encoding='utf-8') as f:
    f.write(lgb_model)

end = time.perf_counter()

print(f'Running time: {end - start} Seconds')

if args.genc:
    import treelite as tl
    import tl2cgen

    cxx_path = f'{prefix}{model_name}{suffix}/'
    libso_path = f'{prefix}{model_name}{suffix}.so'

    tlmodel = tl.frontend.load_lightgbm_model(lgb_path)
    tl2cgen.generate_c_code(tlmodel, dirpath=cxx_path, params={})
    start = time.perf_counter()
    tl2cgen.export_lib(tlmodel, toolchain='gcc', libpath=libso_path, params={})
    end = time.perf_counter()
    print(f'Compile cost: {end - start} Seconds')
