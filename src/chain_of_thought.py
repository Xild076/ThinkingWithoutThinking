import os
import time
import random
import json
import re
import google.generativeai as genai

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


C = type("C", (), {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "HDR": "\033[38;5;45m",
    "PLAN": "\033[38;5;39m",
    "CRIT": "\033[38;5;202m",
    "FIX": "\033[38;5;112m",
    "RESP": "\033[38;5;220m",
    "SCORE": "\033[38;5;213m",
    "IMPROVE": "\033[38;5;119m",
})


def init_model(model_name: str | None = None):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    name = model_name or os.getenv("GENAI_MODEL", "gemma-3-27b-it")
    return genai.GenerativeModel(name)

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def check_rate_limit_budget(prompts: list[str], max_tokens_per_minute: int = 15000) -> float:
    total_estimated_tokens = sum(estimate_tokens(prompt) for prompt in prompts)
    if total_estimated_tokens > max_tokens_per_minute * 0.8:
        return 60.0
    elif total_estimated_tokens > max_tokens_per_minute * 0.6:
        return 30.0
    return 0.0

def plan_prompt(text, verbose=False):
    prompt = (
    f"You are building an internal execution plan to answer the user's prompt.\n"
    f"Return a concise plan for the MODEL to follow, not advice for the user.\n"
    f"No preamble. Keep total under 180 words. Use numbered lists.\n\n"
        f"PROMPT:\n{text}\n\n"
        f"Return exactly:\n"
        f"1) Core Instruction: one-sentence directive capturing the task.\n"
        f"2) Constraints: 3-6 bullets (length, tone, audience, must/avoid).\n"
        f"3) Personality: Construct a personality that would best engage the user and fulfill their needs.\n"
        f"4) Steps: 4-6 numbered actions the MODEL will take. Must include a dedicated final step for verifying all references and recommendations.\n"
        f"5) Evidence To Extract: 3-8 concise items or checks.\n"
        f"6) Rubric: 4-6 pass criteria the final answer must satisfy. Must include a criterion for 'Logical Consistency'.\n"
        f"7) Output Schema: short spec for the final answer structure.\n"
    f"Ensure that within the plan, you are:"
        f"1) answering the core prompt in a way most beneficial to the user's needs, goals, and specific wishes (implied and explicit)"
        f"2) forward looking at potential challenges and obstacles"
        f"3) ensuring feasibility and clarity in the proposed steps."
        f"4) ensuring the final answer is clear, logically coherent, and actionable per the rubric."
        f"5) if the prompt is creative, encouraging imaginative and innovative responses with strong voice and character"
    )
    return generate(prompt, temperature=0.25, max_tokens=400, label="PLAN", verbose=verbose)

def critique_plan_prompt(text, plan, verbose=False):
    prompt = (
        f"You are reviewing an internal execution plan for efficiency and alignment.\n\n"
        f"PROMPT:\n{text}\n\nPLAN:\n{plan}\n\n"
        f"Return exactly 3-6 issues as a numbered list with: issue, impact, fix.\n"
        f"Focus on: weak Core Instruction, inefficient steps, token waste, vague rubric, and especially risks of logical contradiction or factual inconsistency in the final output.\n"
    )
    return generate(prompt, temperature=0.2, max_tokens=300, label="CRITIQUE", verbose=verbose)

def fixer_plan_prompt(text, plan, critique, verbose=False):
    prompt = (
        f"Refine the internal execution plan using the critique.\n\n"
        f"PROMPT:\n{text}\n\nPLAN:\n{plan}\n\nCRITIQUE:\n{critique}\n\n"
        f"Return only PLAN v2 with the same sections:\n"
        f"1) Core Instruction\n2) Constraints\n3) Steps\n4) Evidence To Extract\n5) Rubric\n6) Output Schema\n"
    )
    return generate(prompt, temperature=0.25, max_tokens=500, label="FIX", verbose=verbose)

def executer_prompt(text, plan, verbose=False):
    prompt = (
        f"Execute PLAN v2 precisely to answer the prompt.\n\n"
        f"PROMPT:\n{text}\n\nPLAN v2:\n{plan}\n\n"
        f"Follow Core Instruction, Constraints, Steps, and Output Schema.\n"
        f"Satisfy the Rubric.\n"
        f"Return only the final answer per Output Schema.\n"
    )
    return generate(prompt, temperature=0.6, max_tokens=None, label="EXECUTE", verbose=verbose)

def scorer_prompt(text, response, verbose=False):
    prompt = (
        f"You are a Quality Assurance agent evaluating a response. Evaluate for clarity, logical coherence, and actionability.\n"
        f"Return ONLY in this exact JSON format: {{\"clarity\": <0-10>, \"logic\": <0-10>, \"actionability\": <0-10>, \"feedback\": \"<short feedback <= 200 chars>\"}}\n\n"
        f"SCORING GUIDE:\n"
        f"- clarity: How easy is the response to understand?\n"
        f"- logic: Is the response internally consistent? Are all references correct and do recommendations make logical sense? A single contradiction means a score of 0-2.\n"
        f"- actionability: Can the user act on this feedback effectively?\n\n"
        f"PROMPT:\n{text}\n\nRESPONSE:\n{response}"
    )
    return generate(prompt, temperature=0.1, max_tokens=200, label="SCORE", verbose=verbose)

def improver_prompt(text, response, plan, scorer, verbose=False):
    prompt = (
        f"You are a Quality Control agent. Your primary goal is to fix the CURRENT RESPONSE based on the SCORE FEEDBACK. Pay special attention to low scores in 'logic'.\n"
        f"Review the original PLAN v2 to ensure your fix aligns with the core goals, especially the Rubric's 'Logical Consistency' criterion.\n"
        f"Keep the Output Schema and Constraints intact.\n\n"
        f"PROMPT:\n{text}\n\nPLAN v2:\n{plan}\n\nCURRENT RESPONSE:\n{response}\n\nSCORE FEEDBACK:\n{scorer}\n\n"
        f"Return only the improved final answer.\n"
    )
    return generate(prompt, temperature=0.4, max_tokens=None, label="IMPROVE", verbose=verbose)

def digest_prompt(text, verbose=False):
    prompt = (
        f"Summarize the user's prompt into a compact task digest under 120 words with key requirements, constraints, and desired outcome.\n\nPROMPT:\n{text}\n\nReturn: Task Digest: ..."
    )
    return generate(prompt, temperature=0.2, max_tokens=160, label="DIGEST", verbose=verbose)

def generate(prompt: str, temperature: float = 0.7, max_tokens: int | None = None, top_p: float = 0.95, top_k: int = 40, retries: int = 5, label: str | None = None, verbose: bool = False) -> str:
    model = init_model()
    max_out = max_tokens or 8192
    cfg = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_out, top_p=top_p, top_k=top_k)
    last = None
    start = time.perf_counter()
    if verbose and label:
        print(f"{C.DIM}▶ {label}: requesting...{C.RESET}")
    
    for i in range(retries):
        try:
            resp = model.generate_content(prompt, generation_config=cfg)
            text = getattr(resp, "text", None)
            if not text:
                raise RuntimeError("Empty response")
            if verbose and label:
                dur = time.perf_counter() - start
                print(f"{C.FIX}✓ {label}: done in {dur:.2f}s{C.RESET}")
            return text
        except Exception as e:
            last = e
            error_str = str(e)
            
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                retry_delay = None
                delay_match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
                if delay_match:
                    retry_delay = int(delay_match.group(1))
                else:
                    quota_match = re.search(r'quota_value:\s*(\d+)', error_str)
                    if quota_match:
                        quota_limit = int(quota_match.group(1))
                        retry_delay = 60 if quota_limit > 1000 else 30
                    else:
                        retry_delay = 60
                
                if verbose and label:
                    print(f"{C.CRIT}⚠️  {label}: Rate limit hit. Waiting {retry_delay}s before retry {i+1}/{retries}{C.RESET}")
                
                if i < retries - 1:
                    time.sleep(retry_delay)
                    continue
            
            if i == retries - 1:
                if verbose and label:
                    print(f"{C.CRIT}✗ {label}: failed after {retries} attempts: {error_str[:120]}...{C.RESET}")
                raise
            
            base_delay = 1.5 * (2 ** i) * random.uniform(0.75, 1.25)
            if verbose and label:
                print(f"{C.CRIT}⏳ {label}: backing off {base_delay:.2f}s (attempt {i+1}/{retries}): {error_str[:80]}...{C.RESET}")
            time.sleep(base_delay)
    
    if last:
        raise last
    raise RuntimeError("Generation failed after all retries")

def chain_of_thought(text, print_chain_of_thought=False):
    sep = "=" * 60
    use_digest = len(text) > int(os.getenv("COT_DIGEST_THRESHOLD_CHARS", "1500"))
    ctx = digest_prompt(text, verbose=print_chain_of_thought) if use_digest else text
    
    planned_prompts = [
        f"plan_prompt: {ctx[:200]}...",
        f"critique_plan_prompt: {ctx[:100]}...",
        f"fixer_plan_prompt: {ctx[:100]}...",
        f"executer_prompt: {ctx[:100]}...",
        f"scorer_prompt: {ctx[:100]}...",
    ]
    
    suggested_delay = check_rate_limit_budget(planned_prompts)
    if suggested_delay > 0 and print_chain_of_thought:
        print(f"{C.DIM}⚠️  Estimated high token usage. Adding {suggested_delay}s delay to avoid rate limits...{C.RESET}")
        time.sleep(suggested_delay)
    
    plan = plan_prompt(ctx, verbose=print_chain_of_thought)
    critique = critique_plan_prompt(ctx, plan, verbose=print_chain_of_thought)
    fixed_plan = fixer_plan_prompt(ctx, plan, critique, verbose=print_chain_of_thought)
    response = executer_prompt(ctx, fixed_plan, verbose=print_chain_of_thought)
    score_raw = scorer_prompt(ctx, response, verbose=print_chain_of_thought)
    
    try:
        score_data = json.loads(score_raw)
        clarity = score_data.get("clarity", -1)
        logic = score_data.get("logic", -1)
        actionability = score_data.get("actionability", -1)
        score_feedback = score_data.get("feedback", "No feedback provided")
        composite_score = (clarity * 2 + logic * 4 + actionability * 2) / 8 * 10
    except (json.JSONDecodeError, KeyError):
        try:
            parts = score_raw.split(" | ")
            score_feedback, composite_score = parts[0], int(parts[1])
            clarity = logic = actionability = -1
        except (ValueError, IndexError):
            score_feedback, composite_score = score_raw, -1
            clarity = logic = actionability = -1
    
    improved_response = None
    if composite_score < 90 or (logic != -1 and logic < 6):
        improved_response = improver_prompt(ctx, response, fixed_plan, score_raw, verbose=print_chain_of_thought)
    
    if print_chain_of_thought:
        print(f"\n{C.HDR}{sep}{C.RESET}")
        print(f"{C.BOLD}{C.HDR}CHAIN OF THOUGHT OUTPUT{C.RESET}")
        print(f"{C.HDR}{sep}{C.RESET}\n")
        if use_digest:
            print(f"{C.BOLD}{C.DIM}[DIGEST]{C.RESET}\n{ctx}\n")
        print(f"{C.BOLD}{C.PLAN}[PLAN]{C.RESET}\n{plan}\n")
        print(f"{C.BOLD}{C.CRIT}[CRITIQUE]{C.RESET}\n{critique}\n")
        print(f"{C.BOLD}{C.FIX}[FIXED PLAN]{C.RESET}\n{fixed_plan}\n")
        print(f"{C.BOLD}{C.RESP}[RESPONSE]{C.RESET}\n{response}\n")
        print(f"{C.BOLD}{C.SCORE}[SCORE FEEDBACK]{C.RESET}\n{score_feedback}")
        if clarity != -1:
            print(f"{C.BOLD}{C.SCORE}[SCORES]{C.RESET}: Clarity: {clarity}/10, Logic: {logic}/10, Actionability: {actionability}/10")
        print(f"{C.BOLD}{C.SCORE}[COMPOSITE SCORE]{C.RESET}: {composite_score:.1f}/100\n")
        if improved_response:
            print(f"{C.BOLD}{C.IMPROVE}[IMPROVED RESPONSE]{C.RESET}\n{improved_response}\n")
        print(f"{C.DIM}{sep}{C.RESET}")
        print(f"{C.DIM}END OF OUTPUT{C.RESET}")
        print(f"{C.DIM}{sep}{C.RESET}\n")
    return improved_response if improved_response else response

