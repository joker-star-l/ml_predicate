import onnx
import argparse
from utils import get_attribute
from collections import defaultdict


# default_model = 'nyc-taxi-green-dec-2016_t3_d2_l4_n7_20250204160726_out'
default_model = 'house_16H_d10_l475_n949_20250302095052_out'

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default=default_model)
parser.add_argument('--random_forest', '-rf', action='store_true')
args = parser.parse_args()

if args.random_forest:
    model_path = f'rf_model_output/{args.model}'
else:
    model_path = f'model_output/{args.model}'
model = onnx.load(model_path + '.onnx')

nodes_featureids = list(get_attribute(model, 'nodes_featureids').ints)
node_modes = list(get_attribute(model, 'nodes_modes').strings)

feature_frequency = defaultdict(int)
for i in range(len(nodes_featureids)):
    if node_modes[i] == b'LEAF':
        continue
    feature_frequency[nodes_featureids[i]] += 1

print(f'Feature frequency: {feature_frequency}')

from matplotlib import pyplot as plt
plt.figure()
plt.title("Feature frequency")
plt.bar(
    feature_frequency.keys(), feature_frequency.values(), color="r", align="center"
)
plt.savefig(f"{model_path}_feature_frequency.png")
plt.close()
