import pandas as pd
import numpy as np

data = 'house_16H'
# data = 'nyc-taxi-green-dec-2016'
# data = 'Ailerons'

# TO 10G
data_path = f'./data/{data}_1G.csv'
X = pd.read_csv(data_path)
X2 = X
for i in range(9):
    X2 = pd.concat([X2, X])
    print(X2.shape)
X2.to_csv(f'./data/{data}_10G.csv', index=False)
np.save(f'./data/{data}_10G.npy', X2.values.astype(np.float32))
