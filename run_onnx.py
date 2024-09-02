import numpy as np
import onnxruntime as ort
import time


pruned = False
enable_profiling = False
threads = 4
data = 'house_16H'
scale = '1G'
model = 'house_16H_d10_l405_n809_20240822085650'
func = lambda x: x > 10

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

op.intra_op_num_threads = 1
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])
input_name = ses.get_inputs()[0].name
pred = ses.run([output], {input_name: X})[0]

if enable_profiling:
    ses.end_profiling()
end = time.time()

print(pred)
if not pruned:
    print(f'pred: {func(pred.reshape(-1)).sum()}')
else:
    print(f'pred: {pred.sum()}')
print(f'cost: {end - start}')
