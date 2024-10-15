import numpy as np
import time
import argparse
from sklearn.tree import DecisionTreeRegressor
import joblib
import treelite
import tl2cgen

# python run_treelite.py -m nyc-taxi-green-dec-2016_d10_l859_n1717_20241015054511 -p 13.120699882507319 --pruned 0 -t 1 --toolchain gcc

parser = argparse.ArgumentParser()
parser.add_argument('--threads', '-t', type=int, default=1)
parser.add_argument('--pruned', type=int, default=0)
parser.add_argument('--data', '-d', type=str,default='nyc-taxi-green-dec-2016')
parser.add_argument('--scale', '-s', type=str, default='1G')
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--predicate', '-p', type=float)
parser.add_argument('--toolchain', type=str, default='gcc')
args = parser.parse_args()

pruned = args.pruned
threads = args.threads
data = args.data
scale = args.scale
model = args.model
toolchain = args.toolchain
func = lambda x: x > args.predicate

if pruned == 0:
    mode_path = f'../model/{model}_rf.joblib'
    libpath= f'../model/lib/{model}_rf.so'
    dirpath= f'../model/ccode/{model}_rf/'
elif pruned == 1:
    mode_path = f'../model/{model}_out_rf.joblib'
    libpath= f'../model/lib/{model}_out_rf.so'
    dirpath= f'../model/ccode/{model}_out_rf/'
elif pruned == 2:
    mode_path = f'../model/{model}_out2_rf.joblib'
    libpath= f'../model/lib/{model}_out2_rf.so'
    dirpath= f'../model/ccode/{model}_out2_rf/'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'../data/{data}_{scale}.npy'
X = np.load(data_path)
dmat = tl2cgen.DMatrix(data=X, dtype='float32', missing=None)

costs = []


sklearn_model = joblib.load(mode_path)
model = treelite.sklearn.import_model(sklearn_model)

tl2cgen.export_lib(model, toolchain=toolchain, libpath=libpath)

predictor = tl2cgen.Predictor(libpath)
tl2cgen.generate_c_code(model, dirpath=dirpath,params={})

start = time.perf_counter()

times = 10
for _ in range(times):
    start0 = time.perf_counter()
    pred = predictor.predict(dmat)
    end0 = time.perf_counter()
    costs.append(end0 - start0)

end = time.perf_counter()

costs.sort()
cost = (end - start - costs[0] - costs[-1]) / (times - 2)

print(pred, pred.shape)
if not pruned:
    print(f'pred: {func(pred.reshape(-1)).sum()}')
else:
    print(f'pred: {pred.sum()}')
print(f'cost: {cost}')

with open('result_treelite.csv', 'a', encoding='utf-8') as f:
    f.write(f'{args.model},{pruned},{args.predicate},{data},{scale},{threads},{toolchain},{cost}\n')
