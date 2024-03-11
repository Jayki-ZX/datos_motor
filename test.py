import pandas as pd

def merge_dfs(df1, df2):
    return pd.concat(axis=0, objs=[df1, df2])

def read_motor_csv(name, state):
    df = pd.read_csv(name, \
    names=['row_num', 'sensor_id', 'acc_x', 'acc_y', 'acc_z',\
            'gyr_x', 'gyr_y', 'gyr_z', 'datetime'], sep=';')
    
    df['state'] = state
    df['acc_x'] = df['acc_x'] / 2715
    df['acc_y'] = df['acc_y'] / 2715
    df['acc_z'] = df['acc_z'] / 2715
    df['gyr_x'] = df['gyr_x'] / 2715
    df['gyr_y'] = df['gyr_y'] / 2715
    df['gyr_z'] = df['gyr_z'] / 2715
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

main_df = read_motor_csv('5hz_desbalance.csv', 'unbalanced')
main_df = merge_dfs(main_df, read_motor_csv('7hz.csv', 'balanced'))
main_df = merge_dfs(main_df, read_motor_csv('10hz_desbalan.csv', 'unbalanced'))
main_df = merge_dfs(main_df, read_motor_csv('10hz.csv', 'balanced'))
main_df = merge_dfs(main_df, read_motor_csv('15hz_desbalan.csv', 'unbalanced'))
main_df = merge_dfs(main_df, read_motor_csv('15hz.csv', 'balanced'))
main_df = merge_dfs(main_df, read_motor_csv('20hz_50g.csv', 'balanced')) # este es balanced?
main_df = merge_dfs(main_df, read_motor_csv('20hz_desbalan.csv', 'unbalanced'))
main_df = merge_dfs(main_df, read_motor_csv('20hz.csv', 'balanced'))
main_df = merge_dfs(main_df, read_motor_csv('señal_impulso.csv', 'unbalanced')) # este cuenta como unbalanced?

# print(main_df)
# general data analyisis
import matplotlib.pyplot as plt
from scipy.fft import fft

main_df = main_df.sort_values('datetime')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

curves = ['acc_x', 'acc_y', 'acc_z']
# Iterate over the subplots and plot your data
for i in range(len(curves)):
    ax = axes[i,0]
    ax.plot(main_df['datetime'], main_df[curves[i]])
    ax.set_title(curves[i])

for i in range(len(curves)):
    ax = axes[i,1]
    ax.plot(main_df['datetime'], fft(main_df[curves[i]].values))
    ax.set_title(curves[i])

plt.tight_layout()
plt.show()