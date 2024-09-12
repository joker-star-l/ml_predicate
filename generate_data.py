import pandas as pd
import numpy as np

data = 'Ailerons'
scale_1G = 450
y_label = 'goal'

# TO 1G
data_path = f'./data/{data}.csv'
df = pd.read_csv(data_path)
X = df.drop(y_label, axis=1)
X2 = X
for i in range(scale_1G - 1):
    X2 = pd.concat([X2, X])
    print(X2.shape)
X2.to_csv(f'./data/{data}_1G.csv', index=False)
np.save(f'./data/{data}_1G.npy', X2.values.astype(np.float32))
