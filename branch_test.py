import onnx
import onnxruntime as ort
from tree import Node, TreeEnsembleRegressor
import time
import numpy as np
import sys
sys.setrecursionlimit(1000000)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--left', '-l', action='store_true', default=False)
parser.add_argument('--depth', '-d', type=int, default=10)
args = parser.parse_args()

left = args.left
depth = args.depth

def create_left_tree(node: 'Node', i: int, depth: int):
    if i == depth:
        return
    
    if i == depth - 1:
        left = Node(
            id=i * 2 + 1,
            feature_id=0,
            mode=b'LEAF',
            value=0,
            target_id=None,
            target_weight=1.0,
            samples=1
        )
    else:
        left = Node(
            id=i * 2 + 1,
            feature_id=node.feature_id,
            mode=b'BRANCH_LEQ',
            value=node.value - 1,
            target_id=None,
            target_weight=None,
            samples=None
        )
    node.left = left
    left.parent = node
    create_left_tree(left, i + 1, depth)

    right = Node(
        id=i * 2 + 2,
        feature_id=0,
        mode=b'LEAF',
        value=0,
        target_id=None,
        target_weight=0.0,
        samples=1
    )
    node.right = right
    right.parent = node

    node.samples = node.left.samples + node.right.samples

def create_right_tree(node: 'Node', i: int, depth: int):
    if i == depth:
        return
    
    if i == depth - 1:
        right = Node(
            id=i * 2 + 1,
            feature_id=0,
            mode=b'LEAF',
            value=0,
            target_id=None,
            target_weight=1.0,
            samples=1
        )
    else:
        right = Node(
            id=i * 2 + 1,
            feature_id=node.feature_id,
            mode=b'BRANCH_LEQ',
            value=node.value + 1,
            target_id=None,
            target_weight=None,
            samples=None
        )
    node.right = right
    right.parent = node
    create_right_tree(right, i + 1, depth)

    left = Node(
        id=i * 2 + 2,
        feature_id=0,
        mode=b'LEAF',
        value=0,
        target_id=None,
        target_weight=0.0,
        samples=1
    )
    node.left = left
    left.parent = node

    node.samples = node.left.samples + node.right.samples

node = Node(
    id=0,
    feature_id=0,
    mode=b'BRANCH_LEQ',
    value=float(depth) if left else 0.0,
    target_id=None,
    target_weight=None,
    samples=None
)
if left:
    create_left_tree(node, 0, depth)
else:
    create_right_tree(node, 0, depth)
regressor = TreeEnsembleRegressor.from_tree(node)
any_model = onnx.load('model/Ailerons_d10_l701_n1401_20240903092934.onnx')
any_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
model = regressor.to_model(any_model)

X = (depth + 2) * (-1 if left else 1) * np.ones((1000000, 1), dtype=np.float32)

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

print(pred.shape)
print(f'pred: {pred.sum()}')
print(f'cost: {cost}')

with open('branch_test.csv', 'a', encoding='utf-8') as f:
    f.write(f'{depth},{left},{cost}\n')
