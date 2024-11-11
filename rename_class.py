import pandas as pd

# data = 'bank-marketing'
# data_path = f'data/{data}_origin.csv'
# out_path = f'data/{data}.csv'
# label = 'Class'
# class_map = {
#     1: 1,
#     2: 0
# }

# data = 'electricity'
# data_path = f'data/{data}_origin.csv'
# out_path = f'data/{data}.csv'
# label = 'class'
# class_map = {
#     'DOWN': 0,
#     'UP': 1
# }

data = 'NASA'
data_path = f'data/{data}_origin.csv'
out_path = f'data/{data}.csv'
label = 'hazardous'
class_map = {
    False: 0,
    True: 1
}

df = pd.read_csv(data_path)
if data == 'NASA':
    df = df.loc[:, ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude', 'hazardous']]
print(df.head())
print(df.tail())
df[label] = df[label].map(class_map)
print(df.head())
print(df.tail())
df.to_csv(out_path, index=False)
