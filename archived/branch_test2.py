from tree import Node, TreeEnsembleRegressor
from typing import List, Tuple
import onnx
import onnxruntime as ort
import numpy as np
import time
import os

leafs = 2 ** 10

class GlobalId:
    def __init__(self):
        self.value = 0

    def get(self):
        self.value += 1
        return self.value

def create_full_tree(leaf_count: int) -> Node:

    if leaf_count not in [2**i for i in range(1, 21)]:
        raise ValueError('leaf_count must be a power of 2')

    global_id = GlobalId()

    nodes: List[Node] = []
    for i in range(leaf_count // 2):
        left = Node(
            id = global_id.get(),
            feature_id = 0,
            mode = b'LEAF',
            value = 0.0,
            target_id = None,
            target_weight = i - 0.1,
            samples = 1
        )

        right = Node(
            id = global_id.get(),
            feature_id = 0,
            mode = b'LEAF',
            value = 0.0,
            target_id = None,
            target_weight = i + 0.1,
            samples = 1
        )

        node = Node(
            id = global_id.get(),
            feature_id = 0,
            mode = b'BRANCH_LEQ',
            value = float(i),
            target_id = None,
            target_weight = None,
            samples = left.samples + right.samples
        )

        node.left = left
        left.parent = node

        node.right = right
        right.parent = node

        nodes.append(node)

    while len(nodes) > 1:
        current_nodes: List[Node] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1]
            node = Node(
                id = global_id.get(),
                feature_id = 0,
                mode = b'BRANCH_LEQ',
                value = (left.value + right.value) / 2,
                target_id = None,
                target_weight = None,
                samples = left.samples + right.samples
            )

            node.left = left
            left.parent = node

            node.right = right
            right.parent = node

            current_nodes.append(node)
        
        nodes = current_nodes

    return nodes[0]


def get_leaf_paths(node: Node, left_n: int, right_n: int, path: str, result: List[Tuple[int, int]]):
    if node.parent is not None:
        if node.parent.left == node:
            left_n += 1
            path += 'L'
        else:
            right_n += 1
            path += 'R'
    
    if node.mode == b'LEAF':
        result.append((left_n, right_n, path))
    else:
        get_leaf_paths(node.left, left_n, right_n, path, result)
        get_leaf_paths(node.right, left_n, right_n, path, result)


# if not os.path.exists(f'full_{leafs}.onnx'):
root = create_full_tree(leafs)
leaf_paths = []
get_leaf_paths(root, 0, 0, '', leaf_paths)
regressor = TreeEnsembleRegressor.from_tree(root)
any_model = onnx.load('model/Ailerons_d10_l701_n1401_20240903092934.onnx')
any_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
model = regressor.to_model(any_model)
# onnx.save(model, f'full_{leafs}.onnx')
# else:
#     model = onnx.load(f'full_{leafs}.onnx')

n = 1000000

range_arr = np.random.randint(0, leafs // 2, 100)
print(range_arr)

ith = 0
for i in range_arr:
# for i in range(leafs // 2):

    print(f'ith: {ith}, i: {i}')
    ith += 1

    for delta in [-0.1, 0.1]:

        X = (i + delta) * np.ones((n, 1), dtype=np.float32)

        costs = []
        start = time.perf_counter()
        op = ort.SessionOptions()
        op.intra_op_num_threads = 1
        ses = ort.InferenceSession(model.SerializeToString(), sess_options=op, providers=['CPUExecutionProvider'])
        input_name = ses.get_inputs()[0].name

        times = 5
        for _ in range(times):
            start0 = time.perf_counter()
            pred = ses.run(['variable'], {input_name: X})[0]
            end0 = time.perf_counter()
            costs.append(end0 - start0)

        end = time.perf_counter()

        costs.sort()
        cost = (sum(costs) - costs[0] - costs[-1]) / (times - 2)

        if i + delta != pred[0]:
            raise ValueError(f'Expected {i + delta}, got {pred[0]}')

        print(pred.shape)
        print(f'pred: {pred[0]} {pred.sum()}')
        print(f'cost: {cost}')

        with open('branch_test2.csv', 'a', encoding='utf-8') as f:
            if delta == -0.1:
                f.write(f'{leafs},{i + delta},{leaf_paths[2 * i][0]},{leaf_paths[2 * i][1]},{cost}\n')
            else:
                f.write(f'{leafs},{i + delta},{leaf_paths[2 * i + 1][0]},{leaf_paths[2 * i + 1][1]},{cost}\n')
