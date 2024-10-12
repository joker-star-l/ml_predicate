import onnx
from typing import List
from utils import get_attribute
import pandas as pd
from tree import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l858_n1715_20241010162144')
parser.add_argument('--pruned', type=int, default=1)
args = parser.parse_args()

model_name: str = args.model
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
featues_path = f'data/features_{model_name[:model_name.find("_d")]}.csv'
sql_path = f'{prefix}{model_name}{suffix}.sql'

def onnx2sql(input_model: onnx.ModelProto, features: List[str]) -> str:
    root = model2tree(input_model, None, 0, None)
    return root.tosql(features)

model = onnx.load(onnx_path)
features = pd.read_csv(featues_path)['features'].tolist()
print(features)
sql = onnx2sql(model, features)
# print(sql)

with open(sql_path, 'w', encoding='utf-8') as f:
    f.write(sql)
