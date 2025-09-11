import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

# data = 'wine_quality'
# model = 'wine_quality_t100_d10_l380_n759_20250327094216'
# label = 'quality'
# predicates = [3, 4, 5, 6, 7, 8, 9]

# data = 'bank-marketing'
# model = 'bank-marketing_t10_d10_l318_n635_20250210050038'
# label = 'Class'
# predicates = [0, 1]

data = 'tpcai-uc08'
model = 'tpcai-uc08_t100_d10_l170_n339_20250327100016'
label = 'trip_type'
predicates = [i for i in range(37)]

model_path = f'rf_model/{model}.joblib'
data_path = f'data/{data}.csv'
output_path = f'rf_model_acc_output/{model}.csv'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('type,index,predicate,value\n')

df = pd.read_csv(data_path, index_col=0)
X = df.drop(columns=[label]).values
y = df[label].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X = X_test
y = y_test

# X = X_train
# y = y_train

# predicates[0] = np.median(y)
# print('median: ', predicates[0])

# predicates[0] = np.mean(y)
# print('mean: ', predicates[0])

# print('ground truth: ', y)

functions = []
for p in predicates:
    functions.append(lambda x, p=p: x == p)

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
    # pi_sum = np.array(sum(predis)) > n * sum(yi) / len(yi)
    # print('dt prediction: ', pi_sum)

    yi = yi.astype(np.float32)
    pi_sum = pi_sum.astype(np.float32)
    predi = predi.astype(np.float32)

    dt_accuracy = accuracy_score(yi, pi_sum)
    dt_precision = precision_score(yi, pi_sum)
    dt_recall = recall_score(yi, pi_sum)
    dt_f1 = f1_score(yi, pi_sum)
    dt_confuse = confusion_matrix(yi, pi_sum)
    dt_mcc = matthews_corrcoef(yi, pi_sum)

    rf_accuracy = accuracy_score(yi, predi)
    rf_precision = precision_score(yi, predi)
    rf_recall = recall_score(yi, predi)
    rf_f1 = f1_score(yi, predi)
    rf_confuse = confusion_matrix(yi, predi)
    rf_mcc = matthews_corrcoef(yi, predi)

    dt_rf_accuracy = accuracy_score(yi, pi_sum)
    dt_rf_precision = precision_score(yi, pi_sum)
    dt_rf_recall = recall_score(yi, pi_sum)
    dt_rf_f1 = f1_score(yi, pi_sum)
    dt_rf_confuse = confusion_matrix(yi, pi_sum)
    dt_rf_mcc = matthews_corrcoef(yi, pi_sum)
    

    print(dt_precision, dt_recall, dt_f1, dt_confuse[1,1], dt_mcc)
    print(rf_precision, rf_recall, rf_f1, rf_confuse[1,1], rf_mcc)
    print(dt_rf_precision, dt_rf_recall, dt_rf_f1, dt_rf_confuse[1,1], dt_rf_mcc)

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'dt_accuracy,{i},{predicates[i]},{dt_accuracy}\n')
        f.write(f'dt_precision,{i},{predicates[i]},{dt_precision}\n')
        f.write(f'dt_recall,{i},{predicates[i]},{dt_recall}\n')
        f.write(f'dt_f1,{i},{predicates[i]},{dt_f1}\n')
        f.write(f'dt_tp,{i},{predicates[i]},{dt_confuse[1,1]}\n')
        f.write(f'dt_mcc,{i},{predicates[i]},{dt_mcc}\n')
        
        f.write(f'rf_accuracy,{i},{predicates[i]},{rf_accuracy}\n')
        f.write(f'rf_precision,{i},{predicates[i]},{rf_precision}\n')
        f.write(f'rf_recall,{i},{predicates[i]},{rf_recall}\n')
        f.write(f'rf_f1,{i},{predicates[i]},{rf_f1}\n')
        f.write(f'rf_tp,{i},{predicates[i]},{rf_confuse[1,1]}\n')
        f.write(f'rf_mcc,{i},{predicates[i]},{rf_mcc}\n')

        f.write(f'dt_rf_accuracy,{i},{predicates[i]},{dt_rf_accuracy}\n')
        f.write(f'dt_rf_precision,{i},{predicates[i]},{dt_rf_precision}\n')
        f.write(f'dt_rf_recall,{i},{predicates[i]},{dt_rf_recall}\n')
        f.write(f'dt_rf_f1,{i},{predicates[i]},{dt_rf_f1}\n')
        f.write(f'dt_rf_tp,{i},{predicates[i]},{dt_rf_confuse[1,1]}\n')
        f.write(f'dt_rf_mcc,{i},{predicates[i]},{dt_rf_mcc}\n')

        f.write(f'true,{i},{predicates[i]},{sum(yi)}\n')


# draw
import matplotlib.pyplot as plt

result_file = f'rf_model_acc_output/{model}.csv'

df = pd.read_csv(result_file)
dt_accuracy = df[df['type'] == 'dt_accuracy']
dt_precision = df[df['type'] == 'dt_precision']
dt_recall = df[df['type'] == 'dt_recall']
dt_f1 = df[df['type'] == 'dt_f1']
dt_tp = df[df['type'] == 'dt_tp']
dt_mcc = df[df['type'] == 'dt_mcc']

rf_accuracy = df[df['type'] == 'rf_accuracy']
rf_precision = df[df['type'] == 'rf_precision']
rf_recall = df[df['type'] == 'rf_recall']
rf_f1 = df[df['type'] == 'rf_f1']
rf_tp = df[df['type'] == 'rf_tp']
rf_mcc = df[df['type'] == 'rf_mcc']

dt_rf_accuracy = df[df['type'] == 'dt_rf_accuracy']
dt_rf_precision = df[df['type'] == 'dt_rf_precision']
dt_rf_recall = df[df['type'] == 'dt_rf_recall']
dt_rf_f1 = df[df['type'] == 'dt_rf_f1']
dt_rf_tp = df[df['type'] == 'dt_rf_tp']
dt_rf_mcc = df[df['type'] == 'dt_rf_mcc']

true_ = df[df['type'] == 'true']

plt.figure(figsize=(15, 10))

# plt.plot(dt_accuracy['predicate'].values, dt_accuracy['value'].values, label='ReTree Accuracy')
# plt.plot(dt_precision['predicate'].values, dt_precision['value'].values, label='ReTree Precision')
# plt.plot(dt_recall['predicate'].values, dt_recall['value'].values, label='ReTree Recall')
# plt.plot(dt_f1['predicate'].values, dt_f1['value'].values, label='ReTree F1')
# plt.plot(dt_tp['predicate'].values, dt_tp['value'].values, label='ReTree TP')
plt.plot(dt_mcc['predicate'].values, dt_mcc['value'].values, label='ReTree MCC')

# plt.plot(rf_accuracy['predicate'].values, rf_accuracy['value'].values, label='Original Accuracy')
# plt.plot(rf_precision['predicate'].values, rf_precision['value'].values, label='Original Precision')
# plt.plot(rf_recall['predicate'].values, rf_recall['value'].values, label='Original Recall')
# plt.plot(rf_f1['predicate'].values, rf_f1['value'].values, label='Original F1')
# plt.plot(rf_tp['predicate'].values, rf_tp['value'].values, label='Original TP')
plt.plot(rf_mcc['predicate'].values, rf_mcc['value'].values, label='ReTree MCC')

# plt.plot(dt_rf_accuracy['predicate'].values, dt_rf_accuracy['value'].values, label='ReTree-Original Accuracy')
# plt.plot(dt_rf_precision['predicate'].values, dt_rf_precision['value'].values, label='ReTree-Original Precision')
# plt.plot(dt_rf_recall['predicate'].values, dt_rf_recall['value'].values, label='ReTree-Original Recall')
# plt.plot(dt_rf_f1['predicate'].values, dt_rf_f1['value'].values, label='ReTree-Original F1')
# plt.plot(dt_rf_tp['predicate'].values, dt_rf_tp['value'].values, label='ReTree-Original TP')
# plt.plot(dt_rf_mcc['predicate'].values, dt_rf_mcc['value'].values, label='ReTree MCC')

# plt.plot(true_['predicate'].values, true_['value'].values, label='P')

plt.title(f'onnxruntime with dataset {data}')
plt.ylabel('accuracy')
plt.xlabel('$\\theta$ (prediction = $\\theta$)')
plt.xticks(predicates)
plt.legend()
# plt.ylim(0)

plt.savefig(f'rf_model_acc_output/{model}.png')
