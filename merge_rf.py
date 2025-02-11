import onnx
import time
import pandas as pd
from typing import List, Tuple, Dict
import argparse
from tree import Node, TreeEnsembleRegressor, model2trees
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='bank-marketing_t3_d2_l4_n7_20250209151527')
args = parser.parse_args()

model_name = args.model

model_path = f'rf_model_output/{model_name}_out.onnx'
samples_list_path = f'rf_model_output/{model_name}_out_node_samples.csv'

model = onnx.load(model_path)
samples_list = pd.read_csv(samples_list_path)['node_samples'].tolist()

start = time.perf_counter()
roots = model2trees(model, samples_list)

# only for debug
def debug_samples(root: 'Node') :
    if root.mode != b'LEAF':
        if root.samples != debug_samples(root.left) + debug_samples(root.right):
            raise ValueError('Samples not match')
    return root.samples

reduced_cost = 0

regressor = TreeEnsembleRegressor.from_trees(roots)
output_model = regressor.to_model(model)
onnx.save_model(output_model, model_path.replace('_out.onnx', '_out2.onnx'))
end = time.perf_counter()

print(f'Elapsed time: {end - start:.6f}s')

# only for debug
branch_samples = 0
for root in roots:
    branch_samples += root.branch_samples()
print(f'Branch samples: {branch_samples}', sum(samples_list))
print(f'Reduced cost: {reduced_cost}')
if branch_samples + reduced_cost != sum(samples_list):
    raise ValueError('Branch samples not match')

# only for debug
for root in roots:
    debug_samples(root)

print(f'Performance: {sum(samples_list) / (sum(samples_list) - reduced_cost)}')
