import numpy as np
import onnxruntime as ort
import time
import argparse

# python run_onnx_rf.py -d house_16H -s 1G -m house_16H_d10_l405_n809_20240903080046 -p 13.120699882507319 --pruned 1 -t 1

parser = argparse.ArgumentParser()
parser.add_argument('--pruned', type=int, default=0)
parser.add_argument('--clf', action='store_true')
parser.add_argument('--clf2reg', action='store_true')
parser.add_argument('--enable_profiling', '-ep', action='store_true')
parser.add_argument('--threads', '-t', type=int, default=4)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--scale', '-s', type=str)
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--predicate', '-p', type=float)
args = parser.parse_args()

pruned = args.pruned
clf = args.clf
clf2reg = args.clf2reg
enable_profiling = args.enable_profiling
threads = args.threads
data = args.data
scale = args.scale
model = args.model

if clf:
    func = lambda x: x == args.predicate
else:
    func = lambda x: x > args.predicate

if pruned == 0:
    if clf and clf2reg:
        mode_path = f'rf_model_output/{model}_reg.onnx'
    else:
        mode_path = f'rf_model/{model}.onnx'
elif pruned == 1:
    mode_path = f'rf_model_output/{model}_out.onnx'
elif pruned == 2:
    mode_path = f'rf_model_output/{model}_out2.onnx'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'data/{data}_{scale}.npy'
X = np.load(data_path)

costs = []
start = time.perf_counter()
op = ort.SessionOptions()
if enable_profiling:
    op.enable_profiling = True

op.intra_op_num_threads = threads
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])
input_name = ses.get_inputs()[0].name
output_name = ses.get_outputs()[0].name

times = 1
for _ in range(times):
    start0 = time.perf_counter()
    pred = ses.run([output_name], {input_name: X})[0]
    end0 = time.perf_counter()
    costs.append(end0 - start0)

if enable_profiling:
    ses.end_profiling()
end = time.perf_counter()

costs = [costs[0]] * 5
costs.sort()
cost = (end - start - costs[0] - costs[-1]) / (times - 2)

print(pred, pred.shape)
if not pruned:
    if clf:
        print(f'pred: {pred.astype(np.int64).sum()}')
    else:
        print(f'pred: {pred.sum()}')
    if clf and not clf2reg:
        print(f'pred_func: {func(pred).sum()}')
    else:
        print(f'pred_func: {func(pred.reshape(-1)).sum()}') # meaningless if tree count > 1
else:
    print(f'pred: {(pred.reshape(-1) > 0.5).sum()}')
print(f'cost: {cost}')

with open('result_rf.csv', 'a', encoding='utf-8') as f:
    f.write(f'{model},{pruned},{args.predicate},{data},{scale},{threads},{cost}\n')
