import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('10hz_debalan.csv', \
    names=['row_num', 'sensor_id', 'acc_x', 'acc_y', 'acc_z',\
            'gyr_x', 'gyr_y', 'gyr_z', 'datetime'], sep=';')

df1['state'] = 'unbalanced'
df1['acc_x'] = df1['acc_x'] / 2715
df1['acc_y'] = df1['acc_y'] / 2715
df1['acc_z'] = df1['acc_z'] / 2715
df1['gyr_x'] = df1['gyr_x'] / 2715
df1['gyr_y'] = df1['gyr_y'] / 2715
df1['gyr_z'] = df1['gyr_z'] / 2715
df1['datetime'] = pd.to_datetime(df1['datetime'])
df1 = df1.drop('datetime', axis=1)

df2 = pd.read_csv('10hz.csv', \
    names=['row_num', 'sensor_id', 'acc_x', 'acc_y', 'acc_z',\
            'gyr_x', 'gyr_y', 'gyr_z', 'datetime'], sep=';')

df2['state'] = 'balanced'
df2['acc_x'] = df2['acc_x'] / 2715
df2['acc_y'] = df2['acc_y'] / 2715
df2['acc_z'] = df2['acc_z'] / 2715
df2['gyr_x'] = df2['gyr_x'] / 2715
df2['gyr_y'] = df2['gyr_y'] / 2715
df2['gyr_z'] = df2['gyr_z'] / 2715
df2['datetime'] = pd.to_datetime(df2['datetime'])
df2 = df2.drop('datetime', axis=1)

df = pd.concat(axis=0, objs=[df1, df2])
print(df)

# fft

# rms

# peaks

# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop('state', axis=1)  # Features
y = df['state']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# svm




# figure, axis = plt.subplots(2, 1)

# axis[0].plot(datetimes, acc_x)
# axis[1].plot(datetimes, df1['acc_x'])
# plt.show()