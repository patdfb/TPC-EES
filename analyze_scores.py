import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('results/codebleu_results.csv')

metrics = [
    'codebleu',
    'ngram_match_score',
    'weighted_ngram_match_score',
    'syntax_match_score',
    'dataflow_match_score',
]

present_metrics = [m for m in metrics if m in df.columns]

if not present_metrics:
    raise ValueError('No CodeBLEU metric columns found in results/codebleu_results.csv')

for metric in present_metrics:
    print("=" * 80)
    print(f"{metric.upper()} SCORES BY MODEL AND OUTPUT TYPE")
    print("=" * 80)

    grouped = df.groupby(['model', 'output'])[metric].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    print(grouped)
    print()

    print("=" * 80)
    print(f"OVERALL {metric.upper()} BY MODEL")
    print("=" * 80)
    model_summary = df.groupby('model')[metric].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    print(model_summary)
    print()

    print("=" * 80)
    print(f"OVERALL {metric.upper()} BY OUTPUT TYPE")
    print("=" * 80)
    output_summary = df.groupby('output')[metric].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    print(output_summary)
    print()

    print("=" * 80)
    print(f"MEAN {metric.upper()} PIVOT TABLE (Model vs Output Type)")
    print("=" * 80)
    pivot = df.pivot_table(values=metric, index='model', columns='output', aggfunc='mean').round(4)
    print(pivot)
    print()

    print("=" * 80)
    print(f"OVERALL AVERAGE {metric.upper()}: {df[metric].mean():.4f}")

print("=" * 80)
