import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import joblib
import onnx
import datetime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


data = 'house_16H'
tree_depth = 5
data_count = 10000
f = lambda x: int(x > 10.5)

data_path = f'data/{data}.csv'
df = pd.read_csv(data_path)
df = df.head(data_count)

X = df.drop(columns=['price'])
y = df['price']
y = y.apply(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

model = DecisionTreeClassifier(max_depth=tree_depth)
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
model_name = f'clf_{data}_d{depth}_l{leaves}_n{node_count}_{now}'
joblib_path = f'model_clf/{model_name}.joblib'
onnx_path = f'model_clf/{model_name}.onnx'

joblib.dump(model, joblib_path)
model_onnx = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, X_train.shape[1]]))], options={id(model): {'zipmap': False}})
onnx.save_model(model_onnx, onnx_path)
