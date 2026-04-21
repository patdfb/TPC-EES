import json
import csv
from codebleu import calc_codebleu
from pathlib import Path
import ctypes

import codebleu.codebleu as codebleu_core
import codebleu.utils as codebleu_utils
from tree_sitter import Language


def _language_from_capsule(capsule_obj):
    """Convert a tree-sitter language PyCapsule into the integer pointer expected by Language()."""
    py_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    py_capsule_get_pointer.restype = ctypes.c_void_p
    py_capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = py_capsule_get_pointer(capsule_obj, b"tree_sitter.Language")
    return Language(ptr)


_original_get_tree_sitter_language = codebleu_utils.get_tree_sitter_language


def _patched_get_tree_sitter_language(lang):
    if lang == "python":
        import tree_sitter_python

        raw_language = tree_sitter_python.language()
        try:
            return Language(raw_language)
        except TypeError as exc:
            if "integer is required" not in str(exc):
                raise
            return _language_from_capsule(raw_language)

    return _original_get_tree_sitter_language(lang)


# calc_codebleu imports get_tree_sitter_language directly from codebleu.codebleu,
# so patch both module references.
codebleu_utils.get_tree_sitter_language = _patched_get_tree_sitter_language
codebleu_core.get_tree_sitter_language = _patched_get_tree_sitter_language


# Load results
data = json.loads(Path("outputs/results.json").read_text())

Path("bleu_score").mkdir(exist_ok=True)

results = {}

for model, rows in data.items():
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print('='*60)
    
    results[model] = {}
    
    # Process each task individually
    for task_idx, row in enumerate(rows):
        # Get the actual task ID from the row, or use the enumeration index as fallback
        task_id = row.get('task_id', row.get('id', task_idx))
        print(f"\n  Task {task_id}:")
        
        # Get reference and outputs for this task
        ref = row["reference_code"]
        ref_file = f"bleu_score/{model}_task{task_id}_ref.txt"
        Path(ref_file).write_text(ref)
        
        outputs = [k for k in row.keys() if k.startswith("output_")]
        results[model][f"task_{task_id}"] = {}
        
        for output_key in sorted(outputs):
            # Get output for this task and extract only the code
            hyp = row[output_key]       # Normalize to single line for sacrebleu (remove extra whitespace/newlines)
            hyp_file = f"bleu_score/{model}_task{task_id}_{output_key}.txt"
            Path(hyp_file).write_text(hyp)
            
            result = calc_codebleu([ref], [hyp], lang="python", weights=(0.25, 0.25, 0.25, 0.25))

            results[model][f"task_{task_id}"][output_key] = result
            print(f"        CodeBLEU: {result['codebleu']:.4f}")
            print(f"        N-gram: {result['ngram_match_score']:.4f}, Syntax: {result['syntax_match_score']:.4f}, Dataflow: {result['dataflow_match_score']:.4f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Prepare data for CSV
csv_rows = []
for model, tasks in results.items():
    for task_key, scores in tasks.items():
        task_num = task_key.split("_")[1]
        for output_key, score_dict in scores.items():
            csv_rows.append({
                'model': model,
                'task': task_num,
                'output': output_key,
                'codebleu': score_dict['codebleu'],
                'ngram_match_score': score_dict['ngram_match_score'],
                'weighted_ngram_match_score': score_dict['weighted_ngram_match_score'],
                'syntax_match_score': score_dict['syntax_match_score'],
                'dataflow_match_score': score_dict['dataflow_match_score']
            })

# Print summary
for model, tasks in results.items():
    print(f"\n{model}:")
    for task_key, scores in tasks.items():
        print(f"  {task_key}:")
        for output_key, score_dict in scores.items():
            print(f"    {output_key}: CodeBLEU={score_dict['codebleu']:.4f}, N-gram={score_dict['ngram_match_score']:.4f}, Syntax={score_dict['syntax_match_score']:.4f}, Dataflow={score_dict['dataflow_match_score']:.4f}")

# Save to CSV
csv_file = Path("results/codebleu_results.csv")
if csv_rows:
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'task', 'output', 'codebleu', 'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to {csv_file}")
