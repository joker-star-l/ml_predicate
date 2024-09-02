import duckdb
from duckdb.typing import FLOAT
import numpy as np
import onnxruntime as ort
import time

data = 'house_16H'
scale = '1G'
thread_duckdb = 1
thread_udf = 1
features = ['P1','P5p1','P6p2','P11p4','P14p9','P15p1','P15p3','P16p2','P18p2','P27p4','H2p2','H8p2','H10p1','H13p1','H18pA','H40p4']

pruned = True
test_only = False
func =  '> 10'
model = 'house_16H_d10_l405_n809_20240822085650'
if not pruned:
    mode_path = f'model/{model}.onnx'
    output = 'variable'
else:
    mode_path = f'model_output/{model}_out.onnx'
    output = 'variable'

duckdb.sql("""
create table house_16H (
P1 FLOAT,
P5p1 FLOAT,
P6p2 FLOAT,
P11p4 FLOAT,
P14p9 FLOAT,
P15p1 FLOAT,
P15p3 FLOAT,
P16p2 FLOAT,
P18p2 FLOAT,
P27p4 FLOAT,
H2p2 FLOAT,
H8p2 FLOAT,
H10p1 FLOAT,
H13p1 FLOAT,
H18pA FLOAT,
H40p4 FLOAT
);
""")

duckdb.sql(f"COPY house_16H FROM 'data/{data}_{scale}.csv';")

op = ort.SessionOptions()
op.intra_op_num_threads = thread_udf
ses = ort.InferenceSession(mode_path, sess_options=op, providers=['CPUExecutionProvider'])

def udf(P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA, H40p4):
    X = np.column_stack([P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA, H40p4])
    if test_only:
        return np.zeros(X.shape[0])
    else:
        input_name = ses.get_inputs()[0].name
        return ses.run([output], {input_name: X})[0].reshape(-1)

duckdb.create_function("udf", udf, [FLOAT] * 16, FLOAT, type='arrow')

duckdb.sql(f"SET threads={thread_duckdb};")

print("Start running!!!")
start = time.time()
if not pruned:
    duckdb.sql(f"explain analyze select * from house_16H where udf(P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA, H40p4) {func};")
else:
    duckdb.sql(f"explain analyze select * from house_16H where udf(P1, P5p1, P6p2, P11p4, P14p9, P15p1, P15p3, P16p2, P18p2, P27p4, H2p2, H8p2, H10p1, H13p1, H18pA, H40p4) {func[0]} 0;")
end = time.time()
print(f"cost: {end-start}")