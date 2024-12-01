import pandas as pd
import numpy as np

# data = 'Ailerons'
# scale_1G = 450
# y_label = 'goal'

# data = 'bank-marketing'
# scale_1G = 3700
# y_label = 'Class'

# data = 'california'
# scale_1G = 623
# y_label = 'price_above_median'

# data = 'electricity'
# scale_1G = 449
# y_label = 'class'

# data = 'credit'
# scale_1G = 1403
# y_label = 'SeriousDlqin2yrs'

# data = 'NASA'
# scale_1G = 180
# y_label = 'hazardous'

data = 'medical_charges'
scale_1G = 230
y_label = 'AverageTotalPayments'

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
