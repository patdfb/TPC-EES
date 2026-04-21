import json
import csv
import subprocess
import re
from pathlib import Path


# Load results
data = json.loads(Path("outputs/results.json").read_text())

Path("sacrebleu_score").mkdir(exist_ok=True)

results = {}

for model, rows in data.items():
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print('='*60)
    
    results[model] = {}
    
    # Process each task individually
    for task_idx, row in enumerate(rows):
        print(f"\n  Task {task_idx}:")
        
        # Get reference and outputs for this task
        ref = ' '.join(row["reference_code"].split())
        ref_file = f"sacrebleu_score/{model}_task{task_idx}_ref.txt"
        Path(ref_file).write_text(ref)
        
        outputs = [k for k in row.keys() if k.startswith("output_")]
        results[model][f"task_{task_idx}"] = {}
        
        for output_key in sorted(outputs):
            # Get output for this task and extract only the code
            hyp = ' '.join(row[output_key].split())        # Normalize to single line for sacrebleu (remove extra whitespace/newlines)
            hyp_file = f"sacrebleu_score/{model}_task{task_idx}_{output_key}.txt"
            Path(hyp_file).write_text(hyp)
            
            # Run sacrebleu command
            try:
                cmd = [
                    "sacrebleu",
                    ref_file,
                    "-i", hyp_file,
                    "-m", "bleu",
                    "-b",
                    "-w", "4"
                ]
                print(f"        REF:  {Path(ref_file).read_text()[:80]}")
                print(f"        HYP:  {Path(hyp_file).read_text()[:80]}")

                print(f"      {output_key}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                score = result.stdout.strip()
                results[model][f"task_{task_idx}"][output_key] = score
                print(f"        Score: {score}")
            except subprocess.CalledProcessError as e:
                print(f"      Error running sacrebleu: {e.stderr}")
                print(f"      STDOUT: {e.stdout}")
            except FileNotFoundError:
                print(f"      sacrebleu not found. Install with: pip install sacrebleu")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Prepare data for CSV
csv_rows = []
for model, tasks in results.items():
    for task_key, scores in tasks.items():
        task_num = task_key.split("_")[1]
        for output_key, score in scores.items():
            csv_rows.append({
                'model': model,
                'task': task_num,
                'output': output_key,
                'bleu_score': score
            })

# Print summary
for model, tasks in results.items():
    print(f"\n{model}:")
    for task_key, scores in tasks.items():
        print(f"  {task_key}:")
        for output_key, score in scores.items():
            print(f"    {output_key}: {score}")

# Save to CSV
csv_file = Path("results/sacrebleu_results.csv")
if csv_rows:
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'task', 'output', 'bleu_score'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to {csv_file}")
