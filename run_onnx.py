import numpy as np
import onnxruntime as ort
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pruned', action='store_true')
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

if not pruned:
    mode_path = f'model/{model}.onnx'
    output = 'variable'
else:
    mode_path = f'model_output/{model}_out.onnx'
    output = 'variable'

data_path = f'data/{data}_{scale}.npy'
X = np.load(data_path)

start = time.time()
op = ort.SessionOptions()
if enable_profiling:
    op.enable_profiling = True

op.intra_op_num_threads = threads
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])
input_name = ses.get_inputs()[0].name

times = 3
for _ in range(times):
    pred = ses.run([output], {input_name: X})[0]

if enable_profiling:
    ses.end_profiling()
end = time.time()

print(pred)
if not pruned:
    print(f'pred: {func(pred.reshape(-1)).sum()}')
else:
    print(f'pred: {pred.sum()}')
print(f'cost: {(end - start) / times}')

with open('result.csv', 'a', encoding='utf-8') as f:
    f.write(f'{model},{pruned},{args.predicate},{data},{scale},{threads},{(end - start) / times}\n')
