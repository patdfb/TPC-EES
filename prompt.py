from llama_cpp import Llama
from codecarbon import EmissionsTracker

MODEL_PATH = "./models/codellama-7b-instruct.Q2_K.gguf"
QUESTION = "What is the capital of Portugal?"
PROMPT = (
"Answer the question as briefly as possible with one word.\n"
f"Q: {QUESTION}\n"
"A: "
)

def main():
    # Load the model
    llm = Llama(
        model_path=MODEL_PATH, # Path to the .gguf model file
        n_ctx=2048, # Context window size
        seed=2026, # Fixed random seed for reproducibility
        verbose=False, # Suppress loading logs
    )

    tracker = EmissionsTracker(project_name="LLM_Inference", output_dir=".")
    tracker.start()

    # Run inference
    output = llm(
        prompt=PROMPT,
        max_tokens=10, # One-word answer is enough
        echo=False, # Do not repeat the prompt
        stop=["Q:"], # Stop if the model starts a new question
    )

    emissions = tracker.stop()
    print(f"Energy used (kg CO2): {emissions:.6f}")
    
    # Extract the generated answer
    answer = output["choices"][0]["text"].strip()

    print(f"Question: {QUESTION}")
    print(f"Output: {answer}")

if __name__ == "__main__":
    main()