import matplotlib.pyplot as plt
import pandas as pd
import os

def gen_results_df():
    files = os.listdir('./experimentos')
    main_df = pd.DataFrame(columns=['classifier', 'mean', 'ws', 'sd'])
    for dataset in files:
        temp_df = pd.read_csv('./experimentos/'+dataset, sep=';')
        sd = dataset.split('_')[0]
        ws = dataset.split('_')[1].split('.')[0][1:]
        temp_df['sd'] = sd
        temp_df['ws'] = ws
        main_df = pd.concat((main_df, temp_df), axis=0)
    return main_df

# 'classiffier;mean;ws;sd'
df = gen_results_df()
classifiers = df['classifier'].unique()
window_sizes = [(i+1) * 50 for i in range(32)]
sd = 'fft'

algorithm_data = {}
max_prec_window = '' 
max_prec_classifier = ''
max_prec = 0
for c in classifiers:
    y = []
    for ws in window_sizes:
        # sd
        # this search will be unique and it should be in the same order of the window siez
        prec = df[(df['classifier'] == c) & (df['sd'] == sd) & (df['ws'] == str(ws))]['mean'].values[0]
        y.append(prec)
        if prec >= max_prec:
            max_prec = prec
            max_prec_window = ws
            max_prec_classifier = c
    algorithm_data[c] = y


# Sample data: dictionary where keys are algorithm names and values are lists of precision scores for different window sizes

# Window sizes
plt.figure(figsize=(10, 6))
# Plotting
for algorithm, precision_scores in algorithm_data.items():
    plt.plot(window_sizes, precision_scores, marker='o', label=algorithm)

# Add labels and title
plt.xlabel('Window Size')
plt.ylabel('Precision')
plt.title('Precision Comparison of Algorithms with Varying Window Sizes')
plt.axhline(y=0.95, color='r', linestyle='--', label='Objetivo')
# Add legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
# Show grid
plt.grid(True)
plt.tight_layout()
print(" Max presition {} achieved by {} with window_size {}".format(max_prec, max_prec_classifier, max_prec_window))
# Show the plot
plt.show()