import pandas as pd
import numpy as np
import onnxruntime as ort
import argparse

# python run_onnx.py -d house_16H -m house_16H_d10_l405_n809_20240903080046 -p 13.120699882507319 -t 1

parser = argparse.ArgumentParser()
parser.add_argument('--threads', '-t', type=int, default=4)
parser.add_argument('--data', '-d', type=str, default='nyc-taxi-green-dec-2016')
parser.add_argument('--label', '-l', type=str, default='tipamount')
parser.add_argument('--model', '-m', type=str, default='nyc-taxi-green-dec-2016_d10_l858_n1715_20250103055426')
parser.add_argument('--random_forest', '-rf', action='store_true')
args = parser.parse_args()

threads = args.threads
data = args.data
label = args.label
model = args.model

suffix = '_rf' if args.random_forest else ''
prefix = 'rf_' if args.random_forest else ''

mode_path = f'model{suffix}/{model}.onnx'
data_path = f'data{suffix}/{data}.csv'
out_path = f'{prefix}model/{model}_percent_value.txt'
out_test_path = f'{prefix}model/{model}_percent_value_test.txt'

df = pd.read_csv(data_path)
X = df.drop(columns=[label]).values.astype(np.float32)

op = ort.SessionOptions()
op.intra_op_num_threads = threads
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])
input_name = ses.get_inputs()[0].name
output_name = ses.get_outputs()[0].name

pred = ses.run([output_name], {input_name: X})[0].reshape(-1)
print(pred, pred.shape)

pred_sorted = np.sort(pred)
print(pred_sorted)

percent_value = []
for i in range(1, 101):
    percent_value.append(pred_sorted[int(len(pred_sorted) * i / 100) - 1])

with open(out_path, 'w', encoding='utf-8') as f:
    for pv in percent_value:
        f.write(f'{str(round(pv, 6))}\n')

percent_value_test = []
for i in range(0, 5):
    percent_value_test.append(percent_value[i])
for i in range(94, 99):
    percent_value_test.append(percent_value[i])

with open(out_test_path, 'w', encoding='utf-8') as f:
    for pv in percent_value_test:
        f.write(f'{str(round(pv, 6))}\n')
