import os
import json
import pandas as pd
from tabulate import tabulate
results = {}

for model_name in os.listdir('.'):
    if model_name.startswith('.') or model_name.startswith('..') or model_name.endswith('.py') or model_name.startswith('other_results'):
        continue
    results[model_name] = {}
    for dataset_name in sorted(os.listdir(f'{model_name}/')):
        if model_name.startswith('.') or model_name.startswith('..') or model_name.endswith('.py'):
            continue
        with open(f'{model_name}/{dataset_name}', 'r') as f:
            data = json.load(f)
            accuracy = data['accuracy']
            results[model_name][dataset_name.replace('.json','')] = accuracy

df = pd.DataFrame(results).transpose()
# sort rows by name
df = df.sort_index()
# sort columns by name
df = df[df.columns.sort_values()]
df.to_csv('all_results.csv')
print(tabulate(df, headers='keys', tablefmt='psql'))
