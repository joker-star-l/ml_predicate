#  该代码废弃，旋转难以解决全部的情况，需要使用动态规划

import onnx
import time
import pandas as pd
import argparse
from tree import Node, TreeEnsembleRegressor, model2tree

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l448_n895_20240919101404')
args = parser.parse_args()

model_name = args.model

model_path = f'model_output/{model_name}_out.onnx'
samples_list_path = f'model_output/{model_name}_out_node_samples.csv'


model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

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
