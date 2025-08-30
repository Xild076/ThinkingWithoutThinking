import re
import os
from datasets import load_dataset
from tqdm import tqdm
from chain_of_thought import chain_of_thought, generate

def parse_final_answer(model_output: str) -> float | None:
    if model_output is None:
        return None
    s = str(model_output)
    s = s.replace("```", "")
    s = s.replace("\u2212", "-")
    def _to_float(tok: str) -> float | None:
        t = tok.strip().rstrip(".,;:)")
        t = t.replace("$", "").replace(",", "")
        try:
            return float(t)
        except Exception:
            return None
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    strong = ["final answer", "therefore", "thus", "answer:"]
    medium = ["profit", "total", "result", "final"]
    def last_num_in(text: str) -> float | None:
        toks = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
        return _to_float(toks[-1]) if toks else None
    for ln in reversed(lines):
        if re.fullmatch(r"\$?-?\d[\d,]*(?:\.\d+)?", ln):
            v = _to_float(ln)
            if v is not None:
                return v
    for ln in reversed(lines):
        if "=" in ln:
            v = last_num_in(ln.split("=")[-1])
            if v is not None:
                return v
    for ln in reversed(lines):
        low = ln.lower()
        if any(k in low for k in strong):
            v = last_num_in(ln)
            if v is not None:
                return v
    for ln in reversed(lines):
        low = ln.lower()
        if any(k in low for k in medium):
            v = last_num_in(ln)
            if v is not None:
                return v
    m = re.search(r"(?is)(?:answer|is|are|equals|=)\s*(-?\$?\d[\d,]*(?:\.\d+)?)", s)
    if m:
        v = _to_float(m.group(1))
        if v is not None:
            return v
    tokens = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", s)
    if tokens:
        return _to_float(tokens[-1])
    return None

def build_fewshot_examples(k: int = 8) -> str:
    train = load_dataset("gsm8k", "main")["train"].select(range(k))
    examples = []
    for item in train:
        q = item["question"].strip()
        final = item["answer"].split("####")[-1].strip()
        examples.append(f"Question: {q}\nAnswer: Let's think step by step. The final answer is {final}.")
    return "\n\n".join(examples)

def run_evaluation():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")["test"]
    sample_size = 500
    dataset = dataset.select(range(sample_size))
    fewshot = build_fewshot_examples(8)
    your_system_correct = 0
    baseline_system_correct = 0
    results = []
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        ground_truth_answer_str = item["answer"].split("####")[-1].strip().replace(',', '')
        ground_truth_answer = float(ground_truth_answer_str)
        your_system_output = chain_of_thought(question, print_chain_of_thought=False)
        your_answer = parse_final_answer(your_system_output)
        baseline_prompt = (
            f"{fewshot}\n\nNow answer the new question.\n"
            f"Question: {question}\n"
            f"Answer: Let's think step by step and provide only the final numeric result at the end."
        )
        baseline_output = generate(baseline_prompt, temperature=0.2, verbose=False)
        baseline_answer = parse_final_answer(baseline_output)
        your_correct = your_answer is not None and abs(your_answer - ground_truth_answer) < 1e-3
        baseline_correct = baseline_answer is not None and abs(baseline_answer - ground_truth_answer) < 1e-3
        if your_correct:
            your_system_correct += 1
        if baseline_correct:
            baseline_system_correct += 1
        results.append({
            "question": question,
            "ground_truth": ground_truth_answer,
            "your_system_output": your_system_output,
            "your_system_answer": your_answer,
            "your_system_correct": your_correct,
            "baseline_output": baseline_output,
            "baseline_answer": baseline_answer,
            "baseline_correct": baseline_correct,
        })
    your_accuracy = (your_system_correct / sample_size) * 100
    baseline_accuracy = (baseline_system_correct / sample_size) * 100
    print("\n--- Evaluation Complete ---")
    print(f"Sample Size: {sample_size} questions from GSM8K")
    print(f"Your System's Accuracy: {your_accuracy:.2f}% ({your_system_correct}/{sample_size})")
    print(f"Baseline System's Accuracy: {baseline_accuracy:.2f}% ({baseline_system_correct}/{sample_size})")
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    run_evaluation()