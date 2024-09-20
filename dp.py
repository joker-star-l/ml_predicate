import onnx
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

# TODO
