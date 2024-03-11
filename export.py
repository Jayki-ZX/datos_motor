import pandas as pd
import math
from main import *
from datetime import datetime, timedelta

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

def export_df(df):
    with open('scenario.js', '+w') as f:
        f.write('const data = {\n')
        initial_time = datetime.now()
        cur_time = initial_time
        f.write('accs: [')
        for index, row in df.iterrows():
            f.write("{\"timestamp\":\"")
            time_formatted = cur_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(time_formatted)
            cur_time += timedelta(milliseconds=200)
            f.write("\", \"acc_x\":")
            f.write(str(row['acc_x'])) 
            f.write(", \"acc_y\":")
            f.write(str(row['acc_y']))
            f.write(", \"acc_z\":")
            f.write(str(row['acc_z'])) 
            f.write("},\n")

        f.write('],\n')

        [xf,yf] = fft_from_df(df)
        f.write('fft: {x: ['+',\n'.join(str(x) for x in xf)+'], y: ['+',\n'.join(str(y) for y in yf)+'],},\n')

        f.write('rms: [')

        cur_time = initial_time

        for index, row in df.iterrows():

            f.write("{\"timestamp\":\"")

            time_formatted = cur_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(time_formatted)
            cur_time += timedelta(milliseconds=200)

            f.write("\", \"value\":")
            f.write(str(math.sqrt(row['acc_x']**2 +  row['acc_y']**2 + row['acc_z']**2))) 
            f.write("},\n")

        f.write('],\n')
        f.write('state: "unbalanced"')
        f.write(',\ndescriptor: "5hz_desbalance[4800:7800]"')
        f.write('};\n')
        f.write("\nexport default data;")

export_df(read_motor_csv('5hz_desbalance.csv', 'unbalanced')[4800:7800])