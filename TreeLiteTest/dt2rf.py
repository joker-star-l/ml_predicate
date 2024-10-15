import numpy as np
import argparse
import joblib
from typing import List
import time
import pandas as pd
import copy

from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestRegressor

# python dt2rf.py -d nyc-taxi-green-dec-2016 -m nyc-taxi-green-dec-2016_d10_l859_n1717_20241015054511 --pruned 0

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--data', '-d', type=str, default='nyc-taxi-green-dec-2016')
parser.add_argument('--pruned', type=int, default=0)

args = parser.parse_args()

pruned = args.pruned
model = args.model
data = args.data

if pruned == 0:
    model_path = f'../model/{model}.joblib'
    mode_outputpath = f'../model/{model}_rf.joblib'
elif pruned == 1:
    model_path = f'../model/{model}_out.joblib'
    mode_outputpath = f'../model/{model}_out_rf.joblib'
elif pruned == 2:
    model_path = f'../model/{model}_out2.joblib'
    mode_outputpath = f'../model/{model}_out2_rf.joblib'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'../data/{data}.csv'
dt_model = joblib.load(model_path)

def df2rf(input_model: DecisionTreeRegressor) -> RandomForestRegressor :
    rf = RandomForestRegressor(n_estimators=1, random_state=42)
    data = pd.read_csv(data_path)
    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    rf.fit(X_train, y_train)

    dtree_copy = copy.deepcopy(input_model)

    rf.estimators_ = [dtree_copy]
    rf.n_features_ = dtree_copy.tree_.n_features
    rf.n_outputs_ = dtree_copy.tree_.n_outputs
    rf.classes_ = getattr(dtree_copy, 'classes_', None)
    return rf

def checkmodel(input_model: DecisionTreeRegressor, model: RandomForestRegressor):
    a = input_model.predict([np.zeros(input_model.tree_.n_features), np.ones(input_model.tree_.n_features)])
    b = model.predict([np.zeros(input_model.tree_.n_features), np.ones(input_model.tree_.n_features)])
    print(a, b)
    if a.sum() - b.sum() > 1e-4:
        raise ValueError('onnx2sklearn failed')

start = time.perf_counter()

rf_model = df2rf(dt_model)
checkmodel(dt_model, rf_model)
joblib.dump(rf_model, mode_outputpath)

end = time.perf_counter()

print(f'Running time: {end - start} Seconds')
