import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import joblib
import onnx
import datetime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import argparse

from utils import get_attribute

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str,  default='house_16H')
parser.add_argument('--tree_depth', '-td', type=int, default=10)
parser.add_argument('--data_count', '-dc', type=int, default=10000)
parser.add_argument('--label', '-l', type=str, default='price')
args = parser.parse_args()

data = args.data
tree_depth = args.tree_depth
data_count = args.data_count
label =args.label

data_path = f'data/{data}.csv'
df = pd.read_csv(data_path)
data_count = min(data_count, df.shape[0])
print(f'data_count: {data_count}')
# TODO
# df = df.sample(n=data_count, random_state=42)

X = df.drop(columns=[label])
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

model = DecisionTreeRegressor(max_depth=tree_depth)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'mse: {mean_squared_error(y_test, y_pred)}')

depth = model.get_depth()
print('depth:', depth)
leaves = model.get_n_leaves()
print('leaves:', leaves)
node_count = model.tree_.node_count
print('nodes:', node_count)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model_name = f'{data}_d{depth}_l{leaves}_n{node_count}_{now}'
joblib_path = f'model/{model_name}.joblib'
onnx_path = f'model/{model_name}.onnx'

joblib.dump(model, joblib_path)
model_onnx = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, X_train.shape[1]]))])
nodes_hitrates = get_attribute(model_onnx, 'nodes_hitrates').floats
for i in range(len(nodes_hitrates)):
    nodes_hitrates[i] = model.tree_.n_node_samples[i]
onnx.save_model(model_onnx, onnx_path)

with open('model/model_name.txt', 'w', encoding='utf-8') as f:
    f.write(f'{model_name}\n')

bucket_num = 10
with open('model/model_leaf_range.txt', 'w', encoding='utf-8') as f:
    leaves = list(set(get_attribute(model_onnx, 'target_weights').floats))
    leaves.sort()
    second_min_leaf = leaves[1]
    second_max_leaf = leaves[-2]
    bucket_size = (second_max_leaf - second_min_leaf) / bucket_num
    print('second_min_leaf:', second_min_leaf, 'second_max_leaf:', second_max_leaf, 'bucket_size:', bucket_size)
    for i in range(bucket_num):
        second_min_leaf += bucket_size
        f.write(str(round(second_min_leaf, 6)) + '\n')
