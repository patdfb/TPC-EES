import json
import time
from llama_cpp import Llama
from codecarbon import EmissionsTracker

# ---------------- CONFIG ----------------
MODELS = {
    "codellama": "./models/codellama-7b-instruct.Q2_K.gguf",
    "meta-llama": "./models/Meta-Llama-3-8B-Instruct-Q2_k.gguf",
}

DATASET_PATH = "./mbpp/sanitized-mbpp.json"
N_TASKS = 20
MAX_TOKENS = 200

# ---------------- LOAD DATA ----------------
def load_mbpp(path):
    data = []
    with open(path, "r") as f:
        data = json.load(f)
    return data


data = load_mbpp(DATASET_PATH)

# ---------------- SPLITS ----------------
evaluation_pool = [t for t in data if 11 <= int(t.get("task_id", -1)) <= 510]
few_shot_tasks  = [t for t in data if int(t.get("task_id", -1)) in [2, 3, 4]]

assert len(few_shot_tasks) == 3, "Few-shot tasks (2,3,4) not found in dataset"
assert len(evaluation_pool) >= N_TASKS, "Not enough tasks in evaluation pool"

evaluation_subset = evaluation_pool[:N_TASKS]

# ---------------- PROMPT BUILDERS ----------------
def format_task_with_solution(task):
    tests = "\n".join(task.get("test_list", []))
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{task['prompt']}\n\n"
        "Your code should pass these tests:\n"
        f"{tests}\n\n"
        "[BEGIN]\n"
        f"{task['code']}\n"
        "[DONE]\n\n"
    )

def build_3shot_prompt(few_shot_tasks, task):
    prompt = ""
    for ex in few_shot_tasks:
        prompt += format_task_with_solution(ex)
    tests = "\n".join(task.get("test_list", []))
    prompt += (
        "You are an expert Python programmer, and here is your task:\n"
        f"{task['prompt']}\n\n"
        "Your code should pass these tests:\n"
        f"{tests}\n\n"
        "[BEGIN]\n"
    )
    return prompt

def build_0shot_prompt(task):
    tests = "\n".join(task.get("test_list", []))
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{task['prompt']}\n\n"
        "Your code should pass these tests:\n"
        f"{tests}\n\n"
        "[BEGIN]\n"
    )

# ---------------- INFERENCE ----------------
def run_inference(llm, prompt, project_name):
    tracker = EmissionsTracker(project_name=project_name, output_dir="./outputs")

    start = time.time()
    tracker.start()

    output = llm(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        echo=False,
        stop=["[DONE]", "[END]"]
    )

    energy = tracker.stop()
    elapsed = time.time() - start

    text = output["choices"][0]["text"].strip()
    return text, energy, elapsed

# ---------------- MAIN LOOP ----------------
all_results = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        seed=2026,
        verbose=False,
    )

    model_results = []

    for i, task in enumerate(evaluation_subset):
        task_id = task.get("task_id")
        print(f"  [{model_name}] Task {i+1}/{N_TASKS} (task_id={task_id})")

        # 0-shot
        prompt_0 = build_0shot_prompt(task)
        code_0, energy_0, time_0 = run_inference(llm, prompt_0, f"{model_name}-0shot")

        # 3-shot
        prompt_3 = build_3shot_prompt(few_shot_tasks, task)
        code_3, energy_3, time_3 = run_inference(llm, prompt_3, f"{model_name}-3shot")

        model_results.append({
            "task_id":        task_id,
            "prompt":         task.get("prompt"),
            "reference_code": task.get("code"),
            "tests":          task.get("test_list"),
            "output_0":       code_0,
            "output_3":       code_3,
            "energy_0":       energy_0,
            "energy_3":       energy_3,
            "time_0":         time_0,
            "time_3":         time_3,
        })

    all_results[model_name] = model_results

    # Liberta memória antes de carregar o próximo modelo
    del llm

# ---------------- SAVE ----------------
with open("./outputs/results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\nResultados guardados em results.json")