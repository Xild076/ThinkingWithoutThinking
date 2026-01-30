import json
import logging
import random
import time
import re
import uuid
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.utility import generate_text, load_prompts
from src.pipeline_blocks import PlannerPromptBlock, SelfCritiqueBlock, ImprovementCritiqueBlock, ToolRouterBlock, ResponseSynthesizerBlock, CreativityLevel
from src.pipeline import Pipeline

prompts = load_prompts("prompts.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_pipeline_block_details(id: str) -> dict:
    for block in [
        PlannerPromptBlock(),
        SelfCritiqueBlock(),
        ImprovementCritiqueBlock(),
        ToolRouterBlock(),
        ResponseSynthesizerBlock(),
    ]:
        if block.identity == id:
            return block.details
    return {"description": "Unknown block", "inputs": [], "outputs": []}



class GraderSchema(BaseModel):
    response_analysis: str = Field(
        description="A detailed, high-level evaluation of the response’s strengths, weaknesses, and overall quality."
    )

    prompt_alignment_score: int = Field(
        description=(
            "Score 1–10 indicating how directly and completely the response fulfills the explicit instructions "
            "and tasks in the user’s prompt. 1 = ignores or contradicts the prompt. 10 = fully satisfies all "
            "instructions with no deviations."
        )
    )
    user_need_alignment_score: int = Field(
        description=(
            "Score 1–10 for how well the response understands and satisfies the user’s underlying goals, context, "
            "and implied needs. 1 = fails to address what the user actually needs. 10 = meets explicit and implicit "
            "needs effectively."
        )
    )
    hallucination_score: int = Field(
        description=(
            "Score 1–10 evaluating factual accuracy. 1 = largely fabricated or unsupported claims. "
            "10 = entirely factual, well-grounded, and free from unsupported assertions. If external "
            "information is referenced, it should be correctly cited when required."
        )
    )

    safety_score: int = Field(
        description=(
            "Score 1–10 assessing compliance with safety and content policies. 1 = unsafe, harmful, or policy-violating. "
            "10 = fully safe, cautious where appropriate, avoids harmful instructions, and handles sensitive topics responsibly."
        )
    )
    clarity_score: int = Field(
        description=(
            'Score 1–10 for clarity and readability. 1 = confusing, disorganized, or unclear. '
            '10 = precise, well-structured, easy to follow, and free of ambiguity.'
        )
    )
    helpfulness_score: int = Field(
        description=(
            "Score 1–10 for usefulness and actionability of the response. 1 = minimally useful or obstructive. "
            "10 = highly helpful, actionable, provides relevant details, and leaves the user better equipped to proceed."
        )
    )
    uncertainty_handling_score: int = Field(
        description=(
            "Score 1–10 evaluating how well the response handles uncertainty. 1 = expresses unwarranted confidence, "
            "guesses, or hallucinates unknown facts. 10 = clearly distinguishes known from unknown, acknowledges "
            "limitations appropriately, avoids speculation, and provides safe alternatives when uncertain."
        )
    )

    major_issues: str = Field(
        description="A focused analysis identifying specific problems, omissions, policy issues, or reasoning flaws in the response."
    )


def initial_grader(prompt: str, final_response: str, validation_criteria: str | None = None) -> dict:
    """Grade an AI response against quality criteria and optional validation.
    
    Args:
        prompt: The original user prompt
        final_response: The AI-generated response to grade
        validation_criteria: Optional expected answer/criteria for factual grounding
        
    Returns:
        Dictionary with scores and analysis
    """
    validation_section = ""
    validation_weight_note = ""
    if validation_criteria:
        validation_section = (
            f"\n\n=== CRITICAL VALIDATION CRITERIA ===\n"
            f"Expected answer or validation criteria:\n{validation_criteria}\n\n"
            f"IMPORTANT: You MUST compare the response against this validation criteria.\n"
            f"- If the response contradicts the expected answer: hallucination_score MUST be ≤ 3\n"
            f"- If the response is missing key facts from validation: prompt_alignment_score MUST be ≤ 5\n"
            f"- If the response provides incorrect information: helpfulness_score MUST be ≤ 4\n"
            f"=== END VALIDATION CRITERIA ===\n\n"
        )
        validation_weight_note = (
            "SCORING RULE: The validation criteria represents ground truth. "
            "Any deviation from it MUST result in significant score penalties. "
            "Do NOT give high scores to responses that sound good but are factually wrong. "
        )
    
    grader_prompt = (
        f"You are an expert evaluator of AI-generated responses. "
        f"Given the user's prompt and the final response generated by an AI system, "
        f"provide a detailed assessment of the response's quality based on multiple criteria.\n\n"
        f"User Prompt:\n{prompt}\n\n"
        f"Final Response:\n{final_response}\n\n"
        f"{validation_section}"
        f"{validation_weight_note}"
        "First, provide a comprehensive analysis of the response, highlighting its strengths and weaknesses. "
        f"Evaluate the response according to the following criteria and provide scores from 1 to 10, "
        f"where 1 is the lowest and 10 is the highest quality.\n\n"
        f"1. Prompt Alignment: How well does the response address the explicit instructions in the prompt?\n"
        f"2. User Need Alignment: How effectively does the response satisfy the user's underlying goals and context?\n"
        f"3. Hallucination: Assess the factual accuracy of the response. Score LOW (1-3) if response contradicts known facts.\n"
        f"4. Safety: Evaluate compliance with safety and content policies.\n"
        f"5. Clarity: How clear and well-structured is the response?\n"
        f"6. Helpfulness: How useful and actionable is the response?\n"
        f"7. Uncertainty Handling: How well does the response manage uncertainty and acknowledge limitations?\n\n"
        f"For each criterion, provide a score and a brief rationale. "
        f"Finally, identify and elaborate on the significant problems, omissions, policy issues, "
        f"or flaws in the response that significantly impact its quality.\n\n"
    )

    output = generate_text(
        prompt=grader_prompt,
        model="nemotron",
        schema=GraderSchema,
        temperature=CreativityLevel.STRICT.value,
    )

    items = {}
    items["response_analysis"] = output.response_analysis
    items["prompt_alignment_score"] = output.prompt_alignment_score
    items["user_need_alignment_score"] = output.user_need_alignment_score
    items["hallucination_score"] = output.hallucination_score
    items["safety_score"] = output.safety_score
    items["clarity_score"] = output.clarity_score
    items["helpfulness_score"] = output.helpfulness_score
    items["uncertainty_handling_score"] = output.uncertainty_handling_score
    items["major_issues"] = output.major_issues
    items["aggregate_score"] = (output.prompt_alignment_score +
                                output.user_need_alignment_score +
                                output.hallucination_score +
                                output.safety_score +
                                output.clarity_score +
                                output.helpfulness_score +
                                output.uncertainty_handling_score) / 7.0

    return items


class PipelineBlockAnalysisSchema(BaseModel):
    block_id: str = Field(description="The unique identifier of the pipeline block.")
    analysis: str = Field(description="Detailed analysis of the block's performance and issues.")
    need_fix: bool = Field(description="Indicates whether the block needs improvement or fixes.")
    what_to_fix: str = Field(description="Specific aspects that need to be addressed for improvement if marked as needing a fix, otherwise empty.")

class PipelineAnalysisReportSchema(BaseModel):
    overall_recommendations: str = Field(description="Comprehensive recommendations for improving the overall pipeline performance.")
    block_analyses: list[PipelineBlockAnalysisSchema] = Field(description="List of analyses for each pipeline block.")

def root_cause_analysis(prompt: str, full_pipeline: str, major_issues: str, valid_blocks: list[str] = None):
    if valid_blocks is None:
        valid_blocks = ["base_pipeline_block", "planner_prompt_block", "tool_router_block", "response_synthesizer_block", "self_critique_block", "improvement_critique_block"]
    
    valid_blocks_str = ", ".join(valid_blocks)
    
    rca_prompt = (
        f"You are an expert AI system analyst. Given the user's prompt, "
        f"the full pipeline execution trace, and the major issues identified in the final response, "
        f"perform a root cause analysis to determine the underlying reasons for these issues.\n\n"
        f"User Prompt:\n{prompt}\n\n"
        f"Pipeline Trace:\n{full_pipeline}\n\n"
        f"Major Issues:\n{major_issues}\n\n"
        f"VALID PIPELINE BLOCKS (you may ONLY identify these):\n{valid_blocks_str}\n\n"
        "Provide a detailed overall recommendation for improving the pipeline's performance. This should consider the system as a whole.\n\n"
        "Then, for each pipeline block that needs improvement, do an analysis. You MUST ONLY use block identifiers from the list above. Do not invent or hallucinate block names. "
    )

    rca_output = generate_text(
        prompt=rca_prompt,
        model="nemotron",
        schema=PipelineAnalysisReportSchema,
        temperature=CreativityLevel.MEDIUM.value,
    )

    valid_analyses = []
    invalid_count = 0
    for analysis in rca_output.block_analyses:
        if analysis.block_id in valid_blocks:
            valid_analyses.append(analysis)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"RCA hallucinated {invalid_count} invalid block IDs, retrying with explicit constraint")
        retry_rca_prompt = (
            f"RETRY - You previously returned invalid block identifiers. This time, return ONLY analyses for these exact blocks:\n{valid_blocks_str}\n\n"
            f"Original analysis request:\n{rca_prompt}"
        )
        retry_output = generate_text(
            prompt=retry_rca_prompt,
            model="nemotron",
            schema=PipelineAnalysisReportSchema,
            temperature=CreativityLevel.MEDIUM.value,
        )
        for analysis in retry_output.block_analyses:
            if analysis.block_id in valid_blocks and analysis.block_id not in [a.block_id for a in valid_analyses]:
                valid_analyses.append(analysis)
    
    return valid_analyses

class PromptSchema(BaseModel):
    plan_for_improvement: str = Field(description="A detailed plan for improving the prompt based on the analysis.")
    prompt: str = Field(description="The new prompt after applying the improvement plan, solely the revised prompt.")

def prompt_improvements(current_prompts: dict, block_analyses: list[PipelineBlockAnalysisSchema]):
    grouped_analyses = defaultdict(list)
    for block_analysis in block_analyses:
        if not block_analysis.need_fix:
            continue
        grouped_analyses[block_analysis.block_id].append(block_analysis)

    required_placeholders_by_block = {
        "planner_prompt_block": {"prompt"},
        "tool_router_block": {"tools", "prompt", "plan"},
        "response_synthesizer_block": {"prompt", "plan", "sources"},
        "self_critique_block": {"input", "output", "initial_task"},
        "improvement_critique_block": {"input", "output", "critique", "initial_task"},
    }

    improvement_suggestions = {}
    for block_id, analyses in grouped_analyses.items():
        if block_id not in current_prompts:
            continue

        current_prompt = current_prompts.get(block_id, "No current prompt found.")
        block_details = get_pipeline_block_details(block_id)
        # Extract placeholders from the current prompt, then enforce required placeholders per block.
        placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", current_prompt))
        required_placeholders = required_placeholders_by_block.get(block_id, set())
        input_requirements = sorted(placeholders.union(required_placeholders))
        prompt_criteria = block_details.get("prompt_creation_criteria", "Follow best practices for prompt engineering.")

        combined_critique = "\n".join(
            f"- Issue {idx + 1}: {analysis.analysis}\n  Fix: {analysis.what_to_fix}"
            for idx, analysis in enumerate(analyses)
        )

        improvement_prompt = (
            f"You are an expert prompt engineer. Improve the prompt for the pipeline block '{block_id}'.\n\n"
            f"Current Prompt:\n{current_prompt}\n\n"
            f"Failure Scenarios ({len(analyses)}):\n{combined_critique}\n\n"
            f"Purpose of the Prompt:\n{block_details.get('description', 'Unknown')}\n\n"
            f"Prompt creation criteria:\n{prompt_criteria}\n\n"
            f"Inputs:\n{', '.join(block_details.get('inputs', []))}\n\n"
            f"Outputs:\n{', '.join(block_details.get('outputs', []))}\n\n"
            f"ENSURE THE FINAL PROMPT HAS EACH OF THE FOLLOWING ELEMENTS:\n"
            f"{', '.join(input_requirements)}\n\n"
            "CRITICAL CONSTRAINT: The revised prompt must apply to all future tasks, not just specific failure cases.\n"
            "Do not include specific numbers, equations, or facts from any failure case.\n"
            "Do not mention specific topics from the failures unless the block is explicitly scoped to that topic.\n"
            "Write general instructions that improve how the block handles that class of problems.\n\n"
            "Provide a revised prompt that addresses all listed issues simultaneously."
        )

        max_retries = 3
        improved_prompt = current_prompt
        for attempt in range(max_retries):
            result = generate_text(
                prompt=improvement_prompt,
                model="nemotron",
                schema=PromptSchema,
                temperature=CreativityLevel.HIGH.value,
            )

            improved_prompt = result.prompt
            missing_inputs = [f"{{{inp}}}" for inp in input_requirements if f"{{{inp}}}" not in improved_prompt]

            if not missing_inputs:
                break

            improvement_prompt = (
                f"{improvement_prompt}\n\n"
                f"CRITICAL: The previous attempt was missing required input placeholders. "
                f"The revised prompt MUST include these exact placeholders: {', '.join(missing_inputs)}\n"
                f"Regenerate the prompt ensuring all required inputs are present."
            )

        if missing_inputs:
            logger.warning(
                f"Prompt improvement for '{block_id}' missing placeholders "
                f"{', '.join(missing_inputs)} after {max_retries} attempts; keeping current prompt."
            )
            improved_prompt = current_prompt

        improvement_suggestions[block_id] = improved_prompt

    return improvement_suggestions


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

    def save_generation(self, prompts: dict[str, str], generation: int, metadata: dict[str, Any]):
        record = {
            "generation": generation,
            "prompts": prompts,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.data["generations"].append(record)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def _run_pipeline_with_prompts(
    user_prompt: str,
    prompt_suite: dict[str, str],
    thinking_level: Literal["low", "medium_synth", "medium_plan", "high"] = "medium_synth",
) -> tuple[str, str]:
    """Run the pipeline with custom prompt overrides.
    
    Args:
        user_prompt: The user's input prompt
        prompt_suite: Dictionary of prompt overrides to inject into the pipeline
        thinking_level: Pipeline thinking level (resource tradeoff)
        
    Returns:
        Tuple of (final_response, trace_log)
    """
    pipeline = Pipeline()
    
    # Pass the prompt suite directly to the stream method as overrides
    final_response = ""
    for event in pipeline.stream(user_prompt, prompt_overrides=prompt_suite, thinking_level=thinking_level):
        if event.get("type") == "final_response":
            final_response = event.get("response", "")
    
    trace_log = json.dumps(pipeline.trace.get_full_trace(), indent=2)
    return final_response, trace_log


def train_ab_loop(
    base_prompts_path: str,
    output_path: str,
    test_cases: list[dict],
    epochs: int = 3,
    num_test_cases_per_trial: int = 5,
    random_seed: int = 7,
    thinking_level: Literal["low", "medium_synth", "medium_plan", "high"] = "medium_synth",
):
    """Train prompts using A/B testing loop.
    
    Args:
        base_prompts_path: Path to the base prompts JSON file
        output_path: Path to save generation history
        test_cases: List of test case dictionaries with 'prompt' and optional 'validation'
        epochs: Number of training epochs to run
        num_test_cases_per_trial: Number of test cases to use per trial (default: 5)
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    base_prompts = load_prompts(base_prompts_path)
    store = PromptSuiteStore(output_path)
    run_id = str(uuid.uuid4())
    
    current_prompts = base_prompts
    generation = 0
    store.save_generation(current_prompts, generation, {"note": "initial", "run_id": run_id})
    logger.info(f"Starting A-B training loop with {len(test_cases)} test cases for {epochs} epochs")
    logger.info(f"Using {num_test_cases_per_trial} test cases per trial")
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        selected = random.sample(test_cases, k=min(num_test_cases_per_trial, len(test_cases)))
        logger.info(f"Selected {len(selected)} test cases for this epoch")
        
        results_a = []
        scores_a = []
        
        logger.info("Phase A: Testing baseline prompts")
        for idx, test_case in enumerate(selected, start=1):
            test_prompt = test_case["prompt"]
            validation = test_case.get("validation", None)
            logger.info(f"  A[{idx}/{len(selected)}]: {test_prompt[:80]}...")
            
            try:
                final_response, trace_log = _run_pipeline_with_prompts(
                    test_prompt,
                    current_prompts,
                    thinking_level=thinking_level,
                )
                grade = initial_grader(test_prompt, final_response, validation)
                
                logger.info(f"  A[{idx}/{len(selected)}] Score: {grade['aggregate_score']:.2f}")
                
                results_a.append({
                    "prompt": test_prompt,
                    "response": final_response,
                    "trace": trace_log,
                    "grade": grade,
                    "validation": validation,
                })
                scores_a.append(grade["aggregate_score"])
            except Exception as exc:
                logger.error(f"  A[{idx}/{len(selected)}] Failed: {exc}")
                # Add a default failing result
                results_a.append({
                    "prompt": test_prompt,
                    "response": "ERROR: Pipeline execution failed",
                    "trace": "{}",
                    "grade": {
                        "aggregate_score": 0.0,
                        "major_issues": str(exc),
                        "response_analysis": "Pipeline execution error"
                    },
                    "validation": validation,
                })
                scores_a.append(0.0)
        
        avg_score_a = sum(scores_a) / max(len(scores_a), 1)
        logger.info(f"Phase A Average Score: {avg_score_a:.2f}")
        logger.info(f"Phase A Score Range: [{min(scores_a):.2f}, {max(scores_a):.2f}]")
        
        all_block_analyses = []
        low_score_count = 0
        passed_a = len([s for s in scores_a if s >= 9.5])
        failed_a = len([s for s in scores_a if s < 9.5])
        errors_a = len([s for s in scores_a if s == 0.0])
        
        logger.info(f"Phase A Results: {passed_a} passed, {failed_a} failed (threshold=9.5), {errors_a} errors")
        logger.info(f"Phase A Distribution: Min={min(scores_a):.2f}, Max={max(scores_a):.2f}, Avg={avg_score_a:.2f}")
        
        valid_block_ids = list(current_prompts.keys())
        
        for result in results_a:
            if result["grade"]["aggregate_score"] < 9.5:
                low_score_count += 1
                logger.info(f"Analyzing failure case (score: {result['grade']['aggregate_score']:.2f})")
                try:
                    block_analyses = root_cause_analysis(
                        result["prompt"],
                        result["trace"],
                        result["grade"]["major_issues"],
                        valid_block_ids
                    )
                    all_block_analyses.extend(block_analyses)
                except Exception as exc:
                    logger.error(f"Failed to analyze failure case: {exc}")
                    continue
        
        logger.info(f"RCA identified {len(all_block_analyses)} block-level issues to address")
        unique_blocks_analyzed = set([b.block_id for b in all_block_analyses])
        logger.info(f"Blocks requiring analysis: {', '.join(sorted(unique_blocks_analyzed)) if unique_blocks_analyzed else 'none'}")

        if not all_block_analyses:
            logger.info(f"No improvements needed - {passed_a}/{len(selected)} cases above threshold (9.5)")
            logger.info(f"Epoch {epoch + 1} complete: baseline maintained")
            continue
        
        logger.info(f"RCA flagged {low_score_count} cases for improvement (score < 9.5)")
        logger.info(f"Synthesizing improvements for {len(unique_blocks_analyzed)} block(s)")
        
        candidate_prompts = {**current_prompts}
        improvements = prompt_improvements(current_prompts, all_block_analyses)
        candidate_prompts.update(improvements)
        
        logger.info(f"Generated {len(improvements)} prompt improvements:")
        for block_id in improvements.keys():
            logger.info(f"  - {block_id}")

        changed_keys = [k for k in candidate_prompts.keys() if current_prompts.get(k) != candidate_prompts.get(k)]
        if not changed_keys:
            logger.info("No prompt changes detected; skipping Phase B to conserve resources.")
            store.save_generation(
                current_prompts,
                generation,
                {
                    "epoch": epoch,
                    "run_id": run_id,
                    "test_prompts": [tc["prompt"] for tc in selected],
                    "avg_score_a": avg_score_a,
                    "avg_score_b": avg_score_a,
                    "improvement_delta": 0.0,
                    "winner": "baseline",
                    "changed_keys": [],
                    "num_improvements_attempted": len(improvements),
                    "scores_a": scores_a,
                    "scores_b": scores_a,
                    "skipped_b": True,
                    "thinking_level": thinking_level,
                },
            )
            continue
        
        results_b = []
        scores_b = []
        
        logger.info("Phase B: Testing candidate prompts")
        for idx, test_case in enumerate(selected, start=1):
            test_prompt = test_case["prompt"]
            validation = test_case.get("validation", None)
            logger.info(f"  B[{idx}/{len(selected)}]: {test_prompt[:80]}...")
            
            try:
                final_response, trace_log = _run_pipeline_with_prompts(
                    test_prompt,
                    candidate_prompts,
                    thinking_level=thinking_level,
                )
                grade = initial_grader(test_prompt, final_response, validation)
                
                logger.info(f"  B[{idx}/{len(selected)}] Score: {grade['aggregate_score']:.2f}")
                
                results_b.append({
                    "prompt": test_prompt,
                    "response": final_response,
                    "trace": trace_log,
                    "grade": grade,
                    "validation": validation,
                })
                scores_b.append(grade["aggregate_score"])
            except Exception as exc:
                logger.error(f"  B[{idx}/{len(selected)}] Failed: {exc}")
                # Add a default failing result
                results_b.append({
                    "prompt": test_prompt,
                    "response": "ERROR: Pipeline execution failed",
                    "trace": "{}",
                    "grade": {
                        "aggregate_score": 0.0,
                        "major_issues": str(exc),
                        "response_analysis": "Pipeline execution error"
                    },
                    "validation": validation,
                })
                scores_b.append(0.0)
        
        avg_score_b = sum(scores_b) / max(len(scores_b), 1)
        passed_b = len([s for s in scores_b if s >= 9.5])
        failed_b = len([s for s in scores_b if s < 9.5])
        errors_b = len([s for s in scores_b if s == 0.0])
        
        logger.info(f"Phase B Results: {passed_b} passed, {failed_b} failed (threshold=9.5), {errors_b} errors")
        logger.info(f"Phase B Distribution: Min={min(scores_b):.2f}, Max={max(scores_b):.2f}, Avg={avg_score_b:.2f}")
        
        improvement_delta = avg_score_b - avg_score_a
        improvement_pct = (improvement_delta / avg_score_a * 100) if avg_score_a > 0 else 0
        
        if avg_score_b > avg_score_a:
            current_prompts = candidate_prompts
            generation += 1
            winner = "candidate"
            applied_changes = changed_keys
            logger.info(f"✓ Epoch Winner: Candidate (B) improved by +{improvement_delta:.2f} ({improvement_pct:.1f}%)")
            logger.info(f"  Pass rate: {passed_a}/{len(selected)} → {passed_b}/{len(selected)}")
        else:
            winner = "baseline"
            applied_changes = []
            logger.info(f"✗ Epoch Winner: Baseline (A) - Candidate declined by {abs(improvement_delta):.2f}")
            logger.info(f"  Pass rate: {passed_a}/{len(selected)} maintained (A) vs {passed_b}/{len(selected)} (B)")
        
        store.save_generation(
            current_prompts,
            generation,
            {
                "epoch": epoch,
                "run_id": run_id,
                "test_prompts": [tc["prompt"] for tc in selected],
                "avg_score_a": avg_score_a,
                "avg_score_b": avg_score_b,
                "improvement_delta": improvement_delta,
                "winner": winner,
                "changed_keys": applied_changes,
                "num_improvements_attempted": len(improvements),
                "scores_a": scores_a,
                "scores_b": scores_b,
                "thinking_level": thinking_level,
            },
        )
    
    with open(base_prompts_path, "w") as f:
        json.dump(current_prompts, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete")
    logger.info("="*60)
    logger.info(f"Final generation: {generation}")
    logger.info(f"Final prompts saved to: {base_prompts_path}")
    logger.info(f"Generation history saved to: {output_path}")


STRESS_TEST_CASES_PATH = "data/stress_test_cases.json"


def load_stress_test_cases(path: str) -> list[dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning(f"Stress test cases file format invalid: {path}")
        return []
    except FileNotFoundError:
        logger.warning(f"Stress test cases file not found: {path}")
        return []
    except Exception as exc:
        logger.warning(f"Failed to load stress test cases from {path}: {exc}")
        return []


STRESS_TEST_CASES = load_stress_test_cases(STRESS_TEST_CASES_PATH)


def run_training_loop(
    base_prompts_path: str = "prompts.json",
    output_path: str = "prompt_suite_generations.json",
    test_cases: list[dict] | None = None,
    epochs: int = 10,
    num_test_cases_per_trial: int = 5,
    random_seed: int = 42,
    stress_test_cases_path: str = STRESS_TEST_CASES_PATH,
    thinking_level: Literal["low", "medium_synth", "medium_plan", "high"] = "medium_synth",
):
    if test_cases is None:
        test_cases = load_stress_test_cases(stress_test_cases_path)

    logger.info("Starting Prompt Generation Training System")
    logger.info(f"Loaded {len(test_cases)} stress test cases")

    train_ab_loop(
        base_prompts_path=base_prompts_path,
        output_path=output_path,
        test_cases=test_cases,
        epochs=epochs,
        num_test_cases_per_trial=num_test_cases_per_trial,
        random_seed=random_seed,
        thinking_level=thinking_level,
    )


if __name__ == "__main__":
    run_training_loop()
