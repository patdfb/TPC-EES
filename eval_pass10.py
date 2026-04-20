import json
import threading
import warnings

with open("./outputs/results_pass10.json") as f:
    data = json.load(f)

# ---------------- HELPERS ----------------

def clean_code(code):
    for marker in ["[END]", "[DONE]", "[TESTS]", "```"]:
        if marker in code:
            code = code.split(marker)[0]
    return code.strip()

def run_tests(code, tests, timeout=5):
    result = [False]

    def target():
        namespace = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(clean_code(code), namespace)
                for test in tests:
                    exec(test, namespace)
            result[0] = True
        except Exception:
            result[0] = False

    t = threading.Thread(target=target)
    t.daemon = True
    t.start()
    t.join(timeout)
    return result[0]

def pass_at_10(outputs, tests):
    """Retorna (pass@10, nº de outputs que passaram)."""
    passed = sum(run_tests(code, tests) for code in outputs)
    return int(passed > 0), passed

# ---------------- EVAL ----------------

summary = {}

for model_name, tasks in data.items():

    summary[model_name] = {}

    for strategy in ["0-shot", "3-shot"]:
        key = "outputs_0" if strategy == "0-shot" else "outputs_3"
        scores = []
        task_details = []

        for task in tasks:
            task_id   = task["task_id"]
            tests     = task["tests"]
            p10, n_passed = pass_at_10(task[key], tests)
            scores.append(p10)

            task_details.append({
                "task_id":      task_id,
                "pass@10":      p10,
                "outputs_passed": n_passed,
                "total_outputs":  10,
            })

        avg = sum(scores) / len(scores)

        summary[model_name][strategy] = {
            "pass@10_avg":   avg,
            "tasks_passed":  sum(scores),
            "tasks_total":   len(scores),
            "per_task":      task_details,
        }

# ---------------- SAVE ----------------
with open("./results/pass10.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n\nResultados guardados em pass10.json")