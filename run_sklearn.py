import numpy as np
import time
import argparse
from sklearn.tree import DecisionTreeRegressor
import joblib

# python run_sklearn.py -d house_16H -s 1G -m house_16H_d10_l405_n809_20240903080046 -p 13.120699882507319 --pruned 0 -t 1

parser = argparse.ArgumentParser()
parser.add_argument('--pruned', type=int, default=0)
parser.add_argument('--threads', '-t', type=int, default=4)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--scale', '-s', type=str)
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--predicate', '-p', type=float)
args = parser.parse_args()

pruned = args.pruned
threads = args.threads
data = args.data
scale = args.scale
model = args.model
func = lambda x: x > args.predicate

if pruned == 0:
    mode_path = f'model/{model}.joblib'
elif pruned == 1:
    mode_path = f'model_output/{model}_out.joblib'
elif pruned == 2:
    mode_path = f'model_output/{model}_out2.joblib'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'data/{data}_{scale}.npy'
X = np.load(data_path)

costs = []
start = time.perf_counter()

skmodel: DecisionTreeRegressor = joblib.load(mode_path)
# TODO mutli-thread

times = 5
for _ in range(times):
    start0 = time.perf_counter()
    pred = skmodel.predict(X)
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

with open('result_sklearn.csv', 'a', encoding='utf-8') as f:
    f.write(f'{model},{pruned},{args.predicate},{data},{scale},{threads},{cost}\n')
