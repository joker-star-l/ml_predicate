import numpy as np
import duckdb
import argparse
import time
import pandas as pd

# python run_sql_duckdb.py -d house_16H -s 1G -m house_16H_d10_l405_n809_20240903080046 -p 13.120699882507319 --pruned 0 -t 1

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
model: str = args.model
func = lambda x: x > args.predicate

if pruned == 0:
    mode_path = f'model/{model}.sql'
elif pruned == 1:
    mode_path = f'model_output/{model}_out.sql'
elif pruned == 2:
    mode_path = f'model_output/{model}_out2.sql'
else:
    raise ValueError('pruned must be 0 or 1 or 2')

data_path = f'data/{data}_{scale}.csv'
featues_path = f'data/features_{data}.csv'

features = pd.read_csv(featues_path)['features'].tolist()

set_threads = f"SET threads={48};"
duckdb.sql(set_threads)

ddl = f"create table t ({', '.join([f'{f} FLOAT' for f in features])});"
print(ddl)
duckdb.sql(ddl)

copy = f"COPY t FROM '{data_path}';"
duckdb.sql(copy)

set_threads = f"SET threads={threads};"
duckdb.sql(set_threads)

with open(mode_path, 'r', encoding='utf-8') as f:
    model_sql = f.read()
select = f"explain analyze SELECT ({model_sql}) FROM t;"

costs = []

times = 5
start = time.perf_counter()

for _ in range(times):
    start0 = time.perf_counter()
    duckdb.sql(select)
    end0 = time.perf_counter()
    costs.append(end0 - start0)

end = time.perf_counter()

costs.sort()
cost = (end - start - costs[0] - costs[-1]) / (times - 2)

print(f'cost: {cost}')

with open('result_sql_duckdb.csv', 'a', encoding='utf-8') as f:
    f.write(f'{model},{pruned},{args.predicate},{data},{scale},{threads},{cost}\n')
