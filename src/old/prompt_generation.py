import json
import random
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.utility import generate_text, load_prompts
from src.pipeline_blocks import PlannerPromptBlock, SelfCritiqueBlock, ImprovementCritiqueBlock, ToolRouterBlock, ResponseSynthesizerBlock


class RootCauseSchema(BaseModel):
    root_causes: list[str] = Field(description="List of likely root causes for failures.")
    fixes: list[str] = Field(description="Proposed fixes mapped to the root causes.")


class PromptUpdateSchema(BaseModel):
    updated_prompts: dict[str, str] = Field(description="Updated prompt text keyed by prompt id.")
    rationale: str = Field(description="Short explanation of changes.")


class ScoreSchema(BaseModel):
    score: int = Field(description="Score from 1-10.")
    rationale: str = Field(description="Short justification.")


class HallucinationSchema(BaseModel):
    risk_score: int = Field(description="Hallucination risk score from 1-10 (10 is high risk).")
    notes: str = Field(description="Short notes about any hallucination risk.")


@dataclass
class PromptSuite:
    prompts: dict[str, str]
    generation: int


class PromptSuiteStore:
    def __init__(self, path: str):
        self.path = path
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {"generations": []}

    def save_generation(self, suite: PromptSuite, metadata: dict[str, Any]):
        record = {
            "generation": suite.generation,
            "prompts": suite.prompts,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.data["generations"].append(record)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def _run_pipeline_once(prompt_text: str, suite: dict[str, str]) -> dict[str, Any]:
    planner = PlannerPromptBlock()
    self_critique = SelfCritiqueBlock()
    improvement_critique = ImprovementCritiqueBlock()
    tool_router = ToolRouterBlock()
    synthesizer = ResponseSynthesizerBlock()

    planner.prompt = suite.get("planner_prompt_block", planner.prompt)
    self_critique.prompt = suite.get("self_critique_block", self_critique.prompt)
    improvement_critique.prompt = suite.get("improvement_critique_block", improvement_critique.prompt)
    tool_router.prompt = suite.get("tool_router_block", tool_router.prompt)
    synthesizer.prompt = suite.get("response_synthesizer_block", synthesizer.prompt)

    try:
        plan = planner(prompt_text)
        critique = self_critique(prompt_text, plan, prompt_text)
        improved_plan = improvement_critique(prompt_text, plan, critique, prompt_text)
        routed = tool_router(prompt_text, improved_plan, [])

        response = synthesizer(prompt_text, {"combined": ""}, improved_plan)
        final_critique = self_critique(prompt_text, response, prompt_text)
        final_response = improvement_critique(prompt_text, response, final_critique, prompt_text)
        final_response = _enforce_final_answer(prompt_text, final_response)
    except Exception as e:
        return {
            "error": str(e),
            "final_response": "",
        }

    return {
        "plan": plan,
        "plan_critique": critique,
        "plan_improved": improved_plan,
        "routed": routed,
        "response": response,
        "final_critique": final_critique,
        "final_response": final_response,
    }


def _score_output(question: str, output: str, rubric: str, prompt_text: str) -> ScoreSchema:
    prompt = (
        "Score the output against the rubric. Return a score 1-10 and a short rationale.\n\n"
        f"QUESTION:\n{question}\n\nOUTPUT:\n{output}\n\nPROMPT SUITE (SUMMARY):\n{prompt_text}\n\nRUBRIC:\n{rubric}"
    )
    scored = generate_text(prompt=prompt, model="nemotron", schema=ScoreSchema, temperature=0.2, max_tokens=300)
    if _looks_like_plan_or_critique(output):
        scored.score = min(scored.score, 3)
        scored.rationale = f"{scored.rationale} | Penalized for plan/critique leakage."
    return scored


def _hallucination_score(question: str, output: str) -> HallucinationSchema:
    prompt = (
        "Assess hallucination risk in the output relative to the input. "
        "Return a risk score from 1-10 and brief notes.\n\n"
        f"QUESTION:\n{question}\n\nOUTPUT:\n{output}"
    )
    return generate_text(prompt=prompt, model="nemotron", schema=HallucinationSchema, temperature=0.2, max_tokens=300)


def _looks_like_plan_or_critique(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    markers = [
        "architect's plan",
        "success metrics",
        "target persona",
        "strategy & tools",
        "step-by-step",
        "verdict:",
        "critical issues",
        "minor issues",
        "praise",
    ]
    return any(marker in lower for marker in markers)


def _enforce_final_answer(question: str, output: str) -> str:
    if not output:
        return output
    if not _looks_like_plan_or_critique(output):
        return output

    prompt = (
        "Rewrite the following model output into a direct answer to the user question. "
        "Remove any meta-planning headings, critiques, or system/internal analysis. "
        "Preserve the required format from the question (e.g., bullet count, table) and answer concisely.\n\n"
        f"QUESTION:\n{question}\n\nMODEL OUTPUT:\n{output}"
    )
    try:
        cleaned = generate_text(prompt=prompt, model="gemma", temperature=0.2, max_tokens=600)
        return cleaned if isinstance(cleaned, str) else output
    except Exception:
        return output


def _root_cause(process_log: dict[str, Any]) -> RootCauseSchema:
    prompt = (
        "You are doing root cause analysis of pipeline failures.\n"
        "Identify likely root causes in the prompts/pipeline and propose fixes. "
        "Do not answer the user tasks; focus only on prompt/pipeline issues.\n\n"
        f"PROCESS LOG:\n{json.dumps(process_log, indent=2)[:4000]}"
    )
    return generate_text(prompt=prompt, model="nemotron", schema=RootCauseSchema, temperature=0.3, max_tokens=800)


def _propose_updates(base_prompts: dict[str, str], root_cause: RootCauseSchema) -> PromptUpdateSchema:
    prompt = (
        "You are improving a prompt suite based on root causes.\n"
        "Return updated prompts for any affected blocks.\n\n"
        f"ROOT CAUSES:\n{root_cause.root_causes}\n\n"
        f"FIXES:\n{root_cause.fixes}\n\n"
        f"CURRENT PROMPTS:\n{json.dumps(base_prompts, indent=2)[:4000]}"
    )
    return generate_text(prompt=prompt, model="nemotron", schema=PromptUpdateSchema, temperature=0.5, max_tokens=1200)


def train_prompt_suite(
    base_prompts_path: str,
    output_path: str,
    stress_tests: list[str],
    rubric: str,
    epochs: int = 3,
    random_seed: int = 7,
):
    random.seed(random_seed)
    base_prompts = load_prompts(base_prompts_path)
    store = PromptSuiteStore(output_path)

    suite = PromptSuite(prompts=base_prompts, generation=0)
    store.save_generation(suite, {"note": "initial"})

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        guardrails = [
            "Write a 7-bullet action list for improving study habits for a college student.",
            "Generate a concise troubleshooting checklist for a server that intermittently returns 502 errors after a deploy.",
            "Summarize the key themes of this short surreal story: 'A man wakes in a train station where clocks melt into the floor. He asks a woman at a clock tower for the time, and she responds by singing his childhood nickname. Each time he boards a train, the scenery repeats with small changes, until he finds a mirror that shows a different face.'",
        ]
        selected = random.sample(stress_tests, k=min(3, len(stress_tests)))
        for g in guardrails:
            if g in stress_tests and g not in selected and len(selected) < min(5, len(stress_tests)):
                selected.append(g)
        run_a = []
        for idx, q in enumerate(selected, start=1):
            print(f"  A: Running {idx}/{len(selected)} -> {q[:80]}")
            run_a.append(_run_pipeline_once(q, suite.prompts))

        scores_a = []
        hall_a = []
        prompt_summary = json.dumps(suite.prompts, indent=2)[:1200]
        for q, run in zip(selected, run_a):
            scores_a.append(_score_output(q, run.get("final_response", ""), rubric, prompt_summary).score)
            hall_a.append(_hallucination_score(q, run.get("final_response", "")).risk_score)
        avg_a = sum(scores_a) / max(len(scores_a), 1)
        avg_hall_a = sum(hall_a) / max(len(hall_a), 1)
        print(f"  A: Avg score {avg_a:.2f}")

        root = _root_cause({"questions": selected, "runs": run_a})
        updates = _propose_updates(suite.prompts, root)
        candidate_prompts = {**suite.prompts, **updates.updated_prompts}

        run_b = []
        for idx, q in enumerate(selected, start=1):
            print(f"  B: Running {idx}/{len(selected)} -> {q[:80]}")
            run_b.append(_run_pipeline_once(q, candidate_prompts))

        scores_b = []
        hall_b = []
        candidate_summary = json.dumps(candidate_prompts, indent=2)[:1200]
        for q, run in zip(selected, run_b):
            scores_b.append(_score_output(q, run.get("final_response", ""), rubric, candidate_summary).score)
            hall_b.append(_hallucination_score(q, run.get("final_response", "")).risk_score)
        avg_b = sum(scores_b) / max(len(scores_b), 1)
        avg_hall_b = sum(hall_b) / max(len(hall_b), 1)
        print(f"  B: Avg score {avg_b:.2f}")

        changed_keys = [k for k in candidate_prompts.keys() if suite.prompts.get(k) != candidate_prompts.get(k)]

        if avg_b > avg_a:
            suite = PromptSuite(prompts=candidate_prompts, generation=suite.generation + 1)
            winner = "candidate"
            applied_changes = changed_keys
        else:
            winner = "base"
            applied_changes = []
        print(f"  Winner: {winner}")

        store.save_generation(
            suite,
            {
                "epoch": epoch,
                "selected": selected,
                "avg_a": avg_a,
                "avg_b": avg_b,
                "avg_hallucination_a": avg_hall_a,
                "avg_hallucination_b": avg_hall_b,
                "winner": winner,
                "changed_prompt_keys": applied_changes,
                "root_causes": root.root_causes,
                "fixes": root.fixes,
                "rationale": updates.rationale,
            },
        )

    with open(base_prompts_path, "w") as f:
        json.dump(suite.prompts, f, indent=2)


def run_training_loop():
    stress_tests = [
        "Write a 5-step plan for launching a small app that helps users track daily water intake.",
        "Summarize the key themes of this short surreal story: 'A man wakes in a train station where clocks melt into the floor. He asks a woman at a clock tower for the time, and she responds by singing his childhood nickname. Each time he boards a train, the scenery repeats with small changes, until he finds a mirror that shows a different face.'",
        "Generate a concise troubleshooting checklist for a server that intermittently returns 502 errors after a deploy.",
        "Draft a short creative paragraph about a clock tower in a dream where time flows backward.",
        "Explain the concept of fractions to a 10-year-old using a pizza example.",
        "Provide a Markdown table comparing 3 project management styles: Agile, Waterfall, and Kanban.",
        "Write a 7-bullet action list for improving study habits for a college student.",
        "Extract key risks from this 3-sentence product description: 'Our wearable monitors heart rate and sleep patterns. It stores data in the cloud and shares insights with a coaching app. Users can invite family members to view their trends.'",
        "Create a brief user persona for a fitness app targeting beginners who dislike gyms.",
        "Summarize this hypothetical research abstract in 2 sentences: 'We propose a lightweight transformer for edge devices that reduces latency by 40% while preserving accuracy on speech tasks. Our approach prunes attention heads and quantizes weights. Experiments on three benchmarks show competitive performance.'",
        "Rewrite this paragraph to be more formal and concise: 'The tool is kind of slow and sometimes it doesn't work right. We should probably fix it soon because people are complaining a lot.'",
        "Generate a short dialogue between two characters with distinct voices: one is a meticulous engineer, the other is an impulsive poet.",
        "List 5 potential failure modes for a machine learning pipeline that ingests user-generated text.",
        "Explain the concept of encryption to a 10-year-old without using technical jargon.",
        "Provide an outline for a 500-word essay on digital privacy for high school students.",
        "Write a haiku about winter and time.",
        "Create a JSON object with keys: title, summary, tags. Use the topic 'Sustainable Travel Tips'.",
        "Give three alternative headlines for a news article about AI safety regulation delays.",
        "Describe the emotional arc of a protagonist in 4 sentences: they start hopeful, face repeated setbacks, lose confidence, then regain purpose.",
        "List 6 acceptance criteria for a login feature with email and password, including error handling and security constraints.",
        "Summarize this fictional policy memo in 3 bullet points: 'We will reduce vendor count by 20% this quarter. Teams must document tool usage by Friday. Savings will fund security upgrades.'",
        "Identify and correct ambiguity in this requirement statement: 'The system should respond quickly and be easy to use.'",
        "Draft a 4-step debugging plan for intermittent API timeouts in production.",
        "Write a concise elevator pitch for a sustainable fashion brand that uses recycled materials.",
        "Translate this technical paragraph into plain English: 'Our service employs eventual consistency with read-repair to reconcile divergent replicas.'",
        "Create a checklist for onboarding a new developer to a Python codebase with CI/CD.",
        "Generate 5 test cases for a password reset flow, including edge cases.",
        "Write a concise explanation of why unit tests matter for a small startup.",
        "Propose 3 metrics to track product engagement for a habit-tracking app.",
        "Summarize this short poem in one sentence without quoting it: 'A leaf tumbles through dusk, counting its shadows, and the night answers in wind.'",
    ]

    rubric = "Clarity, completeness, adherence to format, and task alignment."

    train_prompt_suite(
        base_prompts_path="prompts.json",
        output_path="prompt_suite_generations.json",
        stress_tests=stress_tests,
        rubric=rubric,
        epochs=3,
    )
