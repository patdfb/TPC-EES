import json
from llama_cpp import Llama

# ---------------- CONFIG ----------------
MODELS = {
    "codellama": "./models/codellama-7b-instruct.Q2_K.gguf",
    "meta-llama": "./models/Meta-Llama-3-8B-Instruct-Q2_k.gguf",
}

DATASET_PATH = "./mbpp/sanitized-mbpp.json"

N_TASKS      = 20
N_SAMPLES    = 10    # 10 outputs por task por estratégia
MAX_TOKENS   = 200
TEMPERATURE  = 0.8   # temperatura para gerar outputs variados

# ---------------- LOAD DATA ----------------
def load_mbpp(path):
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

# ---------------- INFERENCE (sem energia, sem tempo) ----------------
def run_inference_n(llm, prompt, n):
    """Gera n outputs para o mesmo prompt."""
    outputs = []
    for k in range(n):
        output = llm(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            echo=False,
            stop=["[DONE]"],
            temperature=TEMPERATURE,
        )
        text = output["choices"][0]["text"].strip()
        outputs.append(text)
    return outputs

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

        # 0-shot: 10 outputs
        prompt_0  = build_0shot_prompt(task)
        outputs_0 = run_inference_n(llm, prompt_0, N_SAMPLES)
        print(f"    0-shot: {N_SAMPLES} outputs gerados")

        # 3-shot: 10 outputs
        prompt_3  = build_3shot_prompt(few_shot_tasks, task)
        outputs_3 = run_inference_n(llm, prompt_3, N_SAMPLES)
        print(f"    3-shot: {N_SAMPLES} outputs gerados")

        model_results.append({
            "task_id":        task_id,
            "prompt":         task.get("prompt"),
            "reference_code": task.get("code"),
            "tests":          task.get("test_list"),
            "outputs_0":      outputs_0,   # lista com 10 strings
            "outputs_3":      outputs_3,   # lista com 10 strings
        })

    all_results[model_name] = model_results

    # Liberta memória antes de carregar o próximo modelo
    del llm

# ---------------- SAVE ----------------
with open("results_pass10.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\nResultados guardados em results_pass10.json")