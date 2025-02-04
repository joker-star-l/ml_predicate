import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import joblib
import onnx
import datetime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import argparse
from typing import Dict, List
import statistics

from utils import get_attribute

parser = argparse.ArgumentParser()

parser.add_argument('--tree_depth', '-td', type=int, default=2)
parser.add_argument('--tree_count', '-tc', type=int, default=3)
parser.add_argument('--data_count', '-dc', type=int, default=10000)

parser.add_argument('--data', '-d', type=str,  default='nyc-taxi-green-dec-2016')
parser.add_argument('--label', '-l', type=str, default='tipamount')

# parser.add_argument('--data', '-d', type=str,  default='house_16H')
# parser.add_argument('--label', '-l', type=str, default='price')

# parser.add_argument('--data', '-d', type=str,  default='Ailerons')
# parser.add_argument('--label', '-l', type=str, default='goal')

# parser.add_argument('--data', '-d', type=str,  default='medical_charges')
# parser.add_argument('--label', '-l', type=str, default='AverageTotalPayments')

args = parser.parse_args()

data = args.data
tree_depth = args.tree_depth
tree_count = args.tree_count
data_count = args.data_count
label = args.label

data_path = f'data/{data}.csv'
df = pd.read_csv(data_path)
data_count = min(data_count, df.shape[0])
print(f'data_count: {data_count}')
# TODO
# df = df.sample(n=data_count, random_state=42)

X = df.drop(columns=[label])
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

model = RandomForestRegressor(n_estimators=tree_count, max_depth=tree_depth, n_jobs=48)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'mse: {mean_squared_error(y_test, y_pred)}')

depth = [model.estimators_[i].get_depth() for i in range(tree_count)]
print('depth:', depth)
leaves = [model.estimators_[i].get_n_leaves() for i in range(tree_count)]
print('leaves:', leaves)
node_count = [model.estimators_[i].tree_.node_count for i in range(tree_count)]
print('nodes:', node_count)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model_name = f'{data}_t{tree_count}_d{sum(depth)//tree_count}_l{sum(leaves)//tree_count}_n{sum(node_count)//tree_count}_{now}'
joblib_path = f'rf_model/{model_name}.joblib'
onnx_path = f'rf_model/{model_name}.onnx'

joblib.dump(model, joblib_path)
model_onnx = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, X_train.shape[1]]))])
nodes_hitrates = get_attribute(model_onnx, 'nodes_hitrates').floats
i = 0
for tree in model.estimators_:
    for j in range(tree.tree_.node_count):
        nodes_hitrates[i] = tree.tree_.n_node_samples[j]
        i += 1
onnx.save_model(model_onnx, onnx_path)

with open('rf_model/model_name.txt', 'w', encoding='utf-8') as f:
    f.write(f'{model_name}\n')

bucket_num = 10
with open('rf_model/model_leaf_range.txt', 'w', encoding='utf-8') as f:
    target_treeids = get_attribute(model_onnx, 'target_treeids').ints
    # IMPORTANT: sklearn 转到 onnx 时, 预测值会除以 tree_count 作为 target_weights
    target_weights = get_attribute(model_onnx, 'target_weights').floats

    leaves_map: Dict[int, list] = {}
    for (i, id) in enumerate(target_treeids):
        if id not in leaves_map:
            leaves_map[id] = []
        leaves_map[id].append(target_weights[i])

    # 等高直方图
    p1_5: List[list] = [[], [], [], [], []]
    p95_99: List[list] = [[], [], [], [], []]
    
    for id in leaves_map.keys():
        leaves_map[id] = list(set(leaves_map[id]))
        leaves_map[id].sort()
        leaves = leaves_map[id]

        bucket_size = len(leaves) / 100
        print('tree:', id, 'bucket_size:', bucket_size)
    
        for i in range(1, 5 + 1):
            p1_5[i-1].append(leaves[int(i * bucket_size)])
        for i in range(95, 99 + 1):
            p95_99[i-95].append(leaves[int(i * bucket_size)])

    # print('p1_5:', p1_5)
    # print('p95_99:', p95_99)

    for p in p1_5:
        f.write(str(round(statistics.mean(p), 6)) + '\n')
    for p in p95_99:
        f.write(str(round(statistics.mean(p), 6)) + '\n')
