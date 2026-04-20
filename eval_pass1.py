import json
import threading
import warnings

with open("./outputs/results.json") as f:
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

# ---------------- EVAL ----------------
summary = {}

for model_name, tasks in data.items():
    summary[model_name] = {}

    for strategy in ["0-shot", "3-shot"]:
        key = "output_0" if strategy == "0-shot" else "output_3"
        scores = []
        task_details = []

        for task in tasks:
            task_id = task["task_id"]
            tests   = task["tests"]
            p1      = int(run_tests(task[key], tests))
            scores.append(p1)

            task_details.append({
                "task_id": task_id,
                "pass@1":  p1,
            })

        summary[model_name][strategy] = {
            "pass@1_avg":   sum(scores) / len(scores),
            "tasks_passed": sum(scores),
            "tasks_total":  len(scores),
            "per_task":     task_details,
        }

# ---------------- SAVE ----------------
with open("./results/pass1.json", "w") as f:
    json.dump(summary, f, indent=4)