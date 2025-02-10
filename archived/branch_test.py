import onnx
import onnxruntime as ort
from tree import Node, TreeEnsembleRegressor, create_left_tree, create_right_tree
import time
import numpy as np
import sys
sys.setrecursionlimit(1000000)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--left', '-l', action='store_true', default=False)
parser.add_argument('--depth', '-d', type=int, default=100)
args = parser.parse_args()

left = args.left
depth = args.depth

real_depth = 100
if real_depth == 0:
    node = Node(
        id=0,
        feature_id=0,
        mode=b'LEAF',
        value=0.0,
        target_id=None,
        target_weight=1.0,
        samples=1
    )
else:
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
        create_left_tree(node, 0, real_depth)
    else:
        create_right_tree(node, 0, real_depth)

regressor = TreeEnsembleRegressor.from_tree(node)
any_model = onnx.load('model/Ailerons_d10_l701_n1401_20240903092934.onnx')
any_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
model = regressor.to_model(any_model)

onnx.save(model, f'{"left" if left else "right"}.onnx')

n = 1000000
if left:
    X = np.ones((n, 1), dtype=np.float32)
else:
    X = depth * np.ones((n, 1), dtype=np.float32)

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
