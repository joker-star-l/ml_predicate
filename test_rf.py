import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split

data = 'nyc-taxi-green-dec-2016'
model = 'nyc-taxi-green-dec-2016_d10_l841_n1682_20250119173117'
label = 'tipamount'
predicates = [i * 0.01 for i in range(0, 360)]

# data = 'house_16H'
# model = 'house_16H_d10_l451_n902_20250119173926'
# label = 'price'
# predicates = [i * 0.01 for i in range(800, 1400)]

# data = 'Ailerons'
# model = 'Ailerons_d10_l715_n1430_20250119174109'
# label = 'goal'
# predicates = [i * 0.00001 for i in range(-320, 0)]

# data = 'medical_charges'
# model = 'medical_charges_d10_l897_n1793_20250119174203'
# label = 'AverageTotalPayments'
# predicates = [i * 0.01 for i in range(800, 1200)]

model_path = f'rf_model/{model}.joblib'
data_path = f'data/{data}.csv'
output_path = f'rf_model_output/{model}.csv'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('type,index,predicate,value\n')

df = pd.read_csv(data_path)
X = df.drop(columns=[label]).values
y = df[label].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X = X_test
y = y_test

# predicates[0] = np.median(y)
# print('median: ', predicates[0])

# predicates[0] = np.mean(y)
# print('mean: ', predicates[0])

# print('ground truth: ', y)

functions = []
for p in predicates:
    functions.append(lambda x, p=p: x > p)

skmodel: RandomForestRegressor = joblib.load(model_path)
n = len(skmodel.estimators_)

pred = skmodel.predict(X)
# print('rf prediction: ', pred)

preds = []
for i, dt in enumerate(skmodel.estimators_):
    p = dt.predict(X)
    # print(f'dt{i} prediction: ', p)
    preds.append(p)

for i, f in enumerate(functions):

    yi = functions[i](y)

    predi = functions[i](pred)
    # print('rf prediction: ', predi)

    predis = []
    for j, p in enumerate(preds):
        pj = functions[i](p)
        predis.append(pj)

    pi_sum = np.array(sum(predis)) > n / 2
    # print('dt prediction: ', pi_sum)

    dt_rf = (predi == pi_sum).sum() / pred.shape[0]
    dt_gd = (pi_sum == yi).sum() / pred.shape[0]
    rf_gd = (predi == yi).sum() / pred.shape[0]

    print('dt_rf', dt_rf)
    print('dt_gd', dt_gd)
    print('rf_gd', rf_gd)

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'dt_rf,{i},{predicates[i]},{dt_rf}\n')
        f.write(f'dt_gd,{i},{predicates[i]},{dt_gd}\n')
        f.write(f'rf_gd,{i},{predicates[i]},{rf_gd}\n')


# draw
import matplotlib.pyplot as plt

result_file = f'rf_model_output/{model}.csv'

df = pd.read_csv(result_file)
dt_rf = df[df['type'] == 'dt_rf']
dt_gd = df[df['type'] == 'dt_gd']
rf_gd = df[df['type'] == 'rf_gd']

# plt.plot(dt_rf['predicate'].values, dt_rf['value'].values, label='dt_rf')
plt.plot(rf_gd['predicate'].values, rf_gd['value'].values, label='original')
plt.plot(dt_gd['predicate'].values, dt_gd['value'].values, label='ours')

# plt.plot(rf_gd['predicate'].values, dt_gd['value'].values - rf_gd['value'].values, label='dt_gd-rf_gd')

plt.title(f'onnxruntime with dataset {data}')
plt.ylabel('accuracy')
plt.xlabel('$\\theta$ (prediction > $\\theta$)')
plt.legend()
# plt.ylim(0)

plt.savefig(f'rf_model_output/{model}.png')
