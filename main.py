from scipy.fft import fft, fftfreq
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random

csvs = ['10hz.csv', 
        '10hz_desbalan.csv', 
        '15hz.csv', 
        '15hz_desbalan.csv', 
        '20hz.csv', 
        '20hz_50g.csv', 
        '20hz_desbalan.csv', 
        '5hz_desbalance.csv', 
        '7hz.csv',
        'señal_impulso.csv']

def merge_dfs(df1, df2):
    return pd.concat(axis=0, objs=[df1, df2])

def str_to_datetime(t):
    return datetime.strptime(t, '%Y-%m-%d %H:%M:%S')

def get_ms_diff(t1, t2):
    td = t2-t1
    return td

def read_my_csv(filename, status):
    df = pd.read_csv(filename, sep=';', names=['row', 's_id', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'datetime'])
    # df = df.drop('s_id', axis=1)
    # df = df.drop('row', axis=1)
    df['acc_x'] = df['acc_x']/2715
    df['acc_y'] = df['acc_y']/2715
    df['acc_z'] = df['acc_z']/2715
    df['datetime'] = [str_to_datetime(v) for v in df['datetime'].values]
    df['state'] = status
    return df

def fft_from_df(df, val, start, end):
    my_df = df.iloc[start:end]
    n = len(my_df)
    t = 1/1600
    y = my_df[val].values
    xf = fftfreq(n, t)[:n//2]
    yf = 2.0/n * abs(fft(y)[:n//2])

    return [xf, yf]

def split_df_by_sensor(df):
    df1 = df.loc[df['s_id'] == 1].copy()
    df1 = df1.sort_values(by='datetime')
    add_nanoseconds(df1)
    normalize_time(df1)
    df2 = df.loc[df['s_id'] == 2].copy()
    df2 = df2.sort_values(by='datetime')
    add_nanoseconds(df2)
    normalize_time(df2)
    return [df1, df2]

def normalize_time(df):
    step_size = 0.000625
    df['timestamp'] = [i * step_size for i in range(len(df))]
    return df

def add_nanoseconds(df):
    delta = timedelta(microseconds=625)
    deltas = np.array([delta * (i % 1600) for i in range(len(df))], dtype='timedelta64')
    df['datetime'] = [v + dt for v, dt in zip(df['datetime'].values, deltas)]
    return df

# takes the df and returns 
# [[sample1], [sample2], ... , [samplen]] , [output1, output2, ... , outputn]
# each sample represents the fft of the acelerations in a window of size step_size, for all axis, concatenated, ex.
# sample = [*fft_y_x, *fft_y_y, *fft_y_z]
# sample should be of shape(1, 3 * len(fft))
def gen_fft_ds(df, step_size):
    dataset = np.zeros((1, 3*(step_size//2)))
    predictions = []
    w_s = 0
    w_e = step_size
    while w_e <= len(df):
        # omiti fx porque como n y t son fijos al calcular la fft, las frecuencias siempre seran las mismas, es decir, fx siempre
        # es igual
        [_, fyx] = fft_from_df(df, 'acc_x', w_s, w_e)
        [_, fyy] = fft_from_df(df, 'acc_y', w_s, w_e)
        [_, fyz] = fft_from_df(df, 'acc_z', w_s, w_e)
        dataset = np.append(dataset, [np.concatenate((fyx, fyy, fyz))], axis=0)

        # predictions.append(output)
        predictions.append(df.iloc[w_s]['state'])
        w_s = w_e
        w_e += step_size
    dataset = dataset[1:]

    return [dataset, predictions]

# takes the df and returns 
# [[sample1], [sample2], ... , [samplen]] , [output1, output2, ... , outputn]
# each sample represents a window of size step_size accelerations concatenated, for example
# sample = [acc_1x, acc1_y, acc1_z, acc2_x, acc2_y, acc2_z , ...]
# sample should be of shape(1, 3 * step_size)
def gen_raw_ds(df, step_size):
    dataset = np.zeros((1, 3*step_size))
    predictions = []
    f_df = df[['acc_x', 'acc_y', 'acc_z']]
    w_s = 0
    w_e = step_size
    while w_e <= len(df):
        accs_windows = f_df.iloc[w_s:w_e]
        dataset = np.append(dataset, [accs_windows.to_numpy().flatten()], axis=0)
        predictions.append(df.iloc[w_s]['state'])
        w_s = w_e
        w_e += step_size
    
    dataset = dataset[1:]
    return [dataset, predictions]

import matplotlib.pyplot as plt

# for csv in csvs:
#     df = read_my_csv(csv)
#     [df1, df2] = split_df_by_sensor(df)

#     # plt.plot(df1['datetime'], df1['acc_z'], color='blue')
#     # plt.title(csv)

#     # [ds, pred] = gen_fft_ds(df1,  10, 'balanced')
#     [ds, pred] = gen_raw_ds(df1, 10, 'balanced')
#     print(len(ds))
#     print(len(pred))

# main_df = read_my_csv('5hz_desbalance.csv', 'unbalanced')
# main_df = merge_dfs(main_df, read_my_csv('7hz.csv', 'balanced'))
# main_df = merge_dfs(main_df, read_my_csv('10hz_desbalan.csv', 'unbalanced'))
# main_df = merge_dfs(main_df, read_my_csv('10hz.csv', 'balanced'))
# main_df = merge_dfs(main_df, read_my_csv('15hz_desbalan.csv', 'unbalanced'))
# main_df = merge_dfs(main_df, read_my_csv('15hz.csv', 'balanced'))
# main_df = merge_dfs(main_df, read_my_csv('20hz_50g.csv', 'balanced')) # este es balanced?
# main_df = merge_dfs(main_df, read_my_csv('20hz_desbalan.csv', 'unbalanced'))
# main_df = merge_dfs(main_df, read_my_csv('20hz.csv', 'balanced'))
# main_df = merge_dfs(main_df, read_my_csv('señal_impulso.csv', 'unbalanced')) # este cuenta como unbalanced?
# [df1, df2] = split_df_by_sensor(main_df)

# final_df = pd.concat(axis=0, objs=(df1,df2))
# final_df.to_csv('final.csv', sep=';')


# ML

#   SVM
from sklearn import svm
svm_classifier = svm.SVC(kernel='linear')
#   SGD
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#   NN
from sklearn.neighbors import KNeighborsClassifier
nn_classifier = KNeighborsClassifier(n_neighbors=11)
#   Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb_classifier = GaussianNB()
#   Decision Trees
from sklearn.tree import DecisionTreeClassifier
dc_classifier = DecisionTreeClassifier()
#   Ensemble
# 	       Boosting
from sklearn.ensemble import HistGradientBoostingClassifier
b_classifier = HistGradientBoostingClassifier(max_iter=100)
#          Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10)
#   Neural network
from sklearn.neural_network import MLPClassifier
nen_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.base import clone
classifiers = [svm_classifier, sgd_classifier,
               nn_classifier, gnb_classifier,
               dc_classifier, b_classifier,
               rf_classifier] # nen_classifier]

data_sample_gens = {'fft': gen_fft_ds, 'accs': gen_raw_ds}

final_df = pd.read_csv('final.csv', sep=';', index_col=0)

wss = [(i+1) * 50 for i in range(32)]
wss.sort(reverse=True)
source_data = 'fft'

for window_size in wss:
    for source_data in ['fft', 'accs']:
        [ds, pred] = data_sample_gens[source_data](final_df, window_size)


        data_fit_start = datetime.now()


        with open('./experimentos/{}_w{}.txt'.format(source_data, window_size), 'w+') as f:
            f.write('classifier;mean\n')
            for c in classifiers:
                f1_scores = np.array([])
                for i in range(10):
                    X_train, X_test, y_train, y_test = train_test_split(ds, pred, test_size=0.2)
                    temp_c = clone(c)
                    temp_c.fit(X_train, y_train)
                    y_pred = temp_c.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    # calcula el f1 score para ambas clases y saca el promedio
                    f1_scores = np.append(f1_scores, f1)
                f.write("{};{}\n".format(temp_c.__class__.__name__ ,  np.mean(f1_scores)))

        print("Data fit for {} with ws {} took ".format(source_data, window_size), datetime.now()-data_fit_start)