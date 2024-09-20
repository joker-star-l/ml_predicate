import numpy as np
import onnxruntime as ort
import time
import argparse

# python run_onnx.py -d house_16H -s 1G -m house_16H_d10_l405_n809_20240903080046 -p 13.120699882507319 --pruned -t 1

parser = argparse.ArgumentParser()
parser.add_argument('--pruned', type=int, default=0)
parser.add_argument('--enable_profiling', '-ep', action='store_true')
parser.add_argument('--threads', '-t', type=int, default=4)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--scale', '-s', type=str)
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--predicate', '-p', type=float)
args = parser.parse_args()

pruned = args.pruned
enable_profiling = args.enable_profiling
threads = args.threads
data = args.data
scale = args.scale
model = args.model
func = lambda x: x > args.predicate

if pruned == 0:
    mode_path = f'model/{model}.onnx'
    output = 'variable'
elif pruned == 1:
    mode_path = f'model_output/{model}_out.onnx'
    output = 'variable'
elif pruned == 2:
    mode_path = f'model_output/{model}_out2.onnx'
    output = 'variable'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'data/{data}_{scale}.npy'
X = np.load(data_path)

start = time.perf_counter()
op = ort.SessionOptions()
if enable_profiling:
    op.enable_profiling = True

op.intra_op_num_threads = threads
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])
input_name = ses.get_inputs()[0].name

times = 5
for _ in range(times):
    pred = ses.run([output], {input_name: X})[0]

if enable_profiling:
    ses.end_profiling()
end = time.perf_counter()

print(pred, pred.shape)
if not pruned:
    print(f'pred: {func(pred.reshape(-1)).sum()}')
else:
    print(f'pred: {pred.sum()}')
print(f'cost: {(end - start) / times}')

with open('result.csv', 'a', encoding='utf-8') as f:
    f.write(f'{model},{pruned},{args.predicate},{data},{scale},{threads},{(end - start) / times}\n')
