import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np

# ==== SETTINGS ====
models = ['SB', 'SBCQ3', 'SBCQ5', 'SBCQ10', 'SBCQ20']
metric = 'DSC'  # or 'HD95'
test_sets = ['SB', 'CQ']
colors = {'SB': 'palegoldenrod', 'CQ': 'paleturquoise'}

# ==== LOAD DATA ====
all_data = []

for model in models:
    for test in test_sets:
        file_name = f'{model}_{test}.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            # Exclude patient CT403
            if 'Patient' in df.columns:
                df = df[df['Patient'] != 'CT403']
            # Remove NaNs and zeros
            df = df[df[metric].notna() & (df[metric] != 0)]
            # For HD95, remove values > 30
            if metric == 'HD95':
                df = df[df[metric] <= 30]
            if not df.empty:
                df['Model'] = model
                df['TestSet'] = test
                df['Patient'] = df['patient'].astype(str)  # Make sure 'patient' is a string
                # Include the Patient column
                all_data.append(df[['Model', 'TestSet', 'Patient', metric]])

if all_data:
    data = pd.concat(all_data, ignore_index=True)
else:
    print("No valid data found to concatenate.")
    exit()

# Define the correct model order
model_order = ['SB', 'SBCQ3', 'SBCQ5', 'SBCQ10', 'SBCQ20']

# Convert the Model column to categorical with the correct order
data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)

# # ==== MEAN AND STD ====
# summary = (
#     data.groupby(['Model', 'TestSet'])[metric]
#     .agg(['mean', 'std'])
#     .reset_index()
#     .rename(columns={'mean': 'Mean', 'std': 'Std'})
# )
# summary['Metric'] = metric
# summary = summary[['Model', 'TestSet', 'Metric', 'Mean', 'Std']]
# summary.to_csv(f'metrics_summary_{metric}.csv', index=False)

# ==== BOX PLOT ====
plt.figure(figsize=(10, 6))

positions = []
labels = []
pos = 0
width = 0.3

for i, model in enumerate(models):
    for j, test in enumerate(test_sets):
        subset = data[(data['Model'] == model) & (data['TestSet'] == test)]
        plt.boxplot(subset[metric], positions=[pos + j*width], widths=width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[test], color=colors[test]),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color=colors[test]),
                    capprops=dict(color=colors[test]))
    positions.append(pos + width / 2)
    labels.append(model)
    pos += 1.2

plt.xticks(positions, labels)
plt.ylabel(metric)
plt.legend([plt.Rectangle((0,0),1,1,color=colors['SB']),
            plt.Rectangle((0,0),1,1,color=colors['CQ'])],
           ['SB test set', 'CQ test set'],
           loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'boxplot_{metric}.png', dpi=300)
plt.close()


plt.figure(figsize=(10, 6))

# Make sure the Patient column is present
print(data.columns)
if 'Patient' not in data.columns:
    print("Column 'Patient' not found in the data.")
else:
    # Plot the lines for each patient
    for patient, group in data.groupby('Patient'):
        # Sort the models to keep the correct order
        group_sorted = group.sort_values('Model')
        plt.plot(
            group_sorted['Model'],
            group_sorted[metric],
            marker='o',
            color='gray',
            alpha=0.5,
            linewidth=1
        )

    # Plot the mean of each model
    mean_performance = data.groupby('Model')[metric].mean().reset_index()
    sns.pointplot(
        data=mean_performance,
        x='Model',
        y=metric,
        color='crimson',
        markers='D',
        scale=1.2,
        errwidth=0,
        alpha=1
    )

    plt.ylabel(metric)
    plt.title(f'Per-patient performance across models ({metric})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'patient_evolution_{metric}.png', dpi=300)
    plt.close()

# ==== T-TEST GRIDS ====
def run_ttest_grid(test_set_label):
    pivot_data = data[data['TestSet'] == test_set_label].pivot(columns='Model', values=metric)
    results = pd.DataFrame(index=models, columns=models)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                results.loc[m1, m2] = np.nan
            else:
                try:
                    a = pivot_data[m1].dropna()
                    b = pivot_data[m2].dropna()
                    stat, p = ttest_rel(a, b)
                    results.loc[m1, m2] = p
                except:
                    results.loc[m1, m2] = np.nan
    return results.astype(float)

for test in test_sets:
    pvals = run_ttest_grid(test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pvals, annot=True, fmt=".5f", cmap='cividis', vmin=0, vmax=1, cbar_kws={'label': 'p-value'})
    plt.title(f'Paired T-test p-values ({test} test set)')
    plt.tight_layout()
    plt.savefig(f'ttest_matrix_{test}_{metric}.png', dpi=300)
    plt.close()