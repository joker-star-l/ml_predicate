import pandas as pd

data = 'bank-marketing'
data_path = f'data/{data}_origin.csv'
out_path = f'data/{data}.csv'
class_map = {
    1: 1,
    2: 0
}

df = pd.read_csv(data_path)
print(df.head())
print(df.tail())
df['Class'] = df['Class'].map(class_map)
print(df.head())
print(df.tail())
df.to_csv(out_path, index=False)
