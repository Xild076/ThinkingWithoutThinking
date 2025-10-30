import json
import logging
import re
from typing import Optional

from .utility import generate_text, python_exec_tool, online_query_tool
from .pdf_utils import extract_text_from_pdf, create_pdf_from_text, create_pdf_from_markdown


logger = logging.getLogger(__name__)


class PipelineBlock:
    """Base class for all pipeline blocks."""
    description = "Base pipeline block"
    
    def __call__(self):
        return NotImplementedError("Subclasses should implement this method.")
    

class PlanCreationBlock(PipelineBlock):
    description = "Generate or refine the execution plan"
    
    def __init__(self, self_critique=True):
        self.self_critique = self_critique

    def __call__(self, text) -> str:
        plan = self.plan_prompt(text)
        if self.self_critique:
            critique = self.critique_plan_prompt(text, plan)
            plan = self.fixer_plan_prompt(text, plan, critique)
        return plan
    
    def plan_prompt(self, text):
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = (
            f"Today's date: {current_date}. Reference this for time-sensitive planning.\n\n"
            f"ROLE: You are an expert execution architect designing an internal plan for an AI system to follow.\n\n"
            f"OBJECTIVE: Create a structured, actionable plan that the MODEL will execute to answer the user's request.\n"
            f"This plan is for the MODEL, not for the user. Be direct, specific, and technically precise.\n\n"
            f"USER REQUEST:\n{text}\n\n"
            f"OUTPUT FORMAT (use exactly these section headers):\n\n"
            f"1) CORE INSTRUCTION\n"
            f"   - One clear sentence capturing the primary task and desired outcome\n"
            f"   - Focus on WHAT to achieve, not HOW (steps come later)\n"
            f"   - Example: 'Generate a comprehensive analysis of X with actionable recommendations'\n\n"
            f"2) CONSTRAINTS\n"
            f"   - List 4-6 specific constraints as bullets\n"
            f"   - Include: length/scope, tone, technical depth, format requirements\n"
            f"   - State what to INCLUDE and what to AVOID\n"
            f"   - Be concrete (e.g., '500-800 words' not 'concise', 'Include 3 examples' not 'provide examples')\n\n"
            f"3) PERSONALITY\n"
            f"   - Define the persona/voice the MODEL should adopt\n"
            f"   - Match the user's needs (e.g., 'expert educator', 'creative storyteller', 'technical analyst')\n"
            f"   - One sentence describing communication style and expertise level\n\n"
            f"4) EXECUTION STEPS\n"
            f"   - List 4-6 sequential actions the MODEL will perform\n"
            f"   - Each step should be specific and measurable\n"
            f"   - ALWAYS include verification as the final step\n"
            f"   - Example format: '1. Analyze X by extracting Y. 2. Generate Z using method M. 3. Verify all claims against provided data.'\n\n"
            f"5) EVIDENCE TO EXTRACT\n"
            f"   - List 4-8 specific data points, facts, or elements the MODEL must identify\n"
            f"   - Be concrete: 'numerical trends in dataset', 'key themes in text', 'visual patterns'\n"
            f"   - These drive block execution (code tools, internet searches, etc.)\n\n"
            f"6) QUALITY RUBRIC\n"
            f"   - Define 4-6 pass/fail criteria for the final answer\n"
            f"   - Make criteria measurable when possible\n"
            f"   - MUST include: 'Logical consistency with zero contradictions or dogma'\n"
            f"   - Examples: 'Contains 3+ concrete examples', 'All data is factually verifiable', 'Directly actionable without additional research'\n\n"
            f"7) OUTPUT SCHEMA\n"
            f"   - Describe the structure of the final answer (as a GUIDE only - actual output is markdown)\n"
            f"   - Specify sections, ordering, and key components\n"
            f"   - Example: 'Intro (1 para) → Analysis (3 sections) → Recommendations (numbered list) → Conclusion'\n\n"
            f"CRITICAL PLANNING PRINCIPLES:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"SCOPE & INTENT:\n"
            f"  • Prioritize user's explicit AND implicit needs - read between the lines\n"
            f"  • Anticipate obstacles and plan mitigation strategies\n"
            f"  • Ensure final output is complete, self-contained, and immediately usable\n\n"
            f"CREATIVE WORK (poems, stories, art, designs):\n"
            f"  • Demand originality - explicitly forbid clichés and obvious interpretations\n"
            f"  • Specify emotional targets: 'evoke wonder', 'create tension', 'inspire reflection'\n"
            f"  • Plan for rich sensory details, unexpected metaphors, and layered meaning\n"
            f"  • Set high quality bars: 'publication-ready', 'gallery-worthy', 'memorable and moving'\n\n"
            f"VISUALIZATIONS (charts, graphs, plots):\n"
            f"  • When user requests visuals: MANDATE code tool usage with matplotlib\n"
            f"  • Specify: plots MUST be saved to files via plt.show() (auto-saved by system)\n"
            f"  • Require: clear labels, titles, legends, and appropriate chart types\n"
            f"  • Plan for synthesis to embed images inline using markdown image syntax\n"
            f"  • If visuals aren't feasible with available data: plan to state this limitation clearly\n\n"
            f"PIPELINE EFFICIENCY:\n"
            f"  • TARGET: 3-5 blocks total (including synthesis)\n"
            f"  • MAXIMUM: 7 blocks (hard limit for API quota)\n"
            f"  • Avoid redundancy - one well-crafted block beats multiple weak ones\n"
            f"  • Only plan multi-block sequences when truly necessary for sequential processing\n\n"
            f"QUALITY & ACCURACY:\n"
            f"  • Build in verification steps - don't trust without checking\n"
            f"  • Explicitly forbid hallucinating data, sources, or images\n"
            f"  • Require date/time accuracy - verify temporal references\n"
            f"  • Demand logical consistency - zero tolerance for contradictions\n"
            f"  • For creative brainstorming: ALWAYS plan to select the best idea for execution\n\n"
            f"DELIVERABLE COMPLETENESS:\n"
            f"  • Final answer must be COMPLETE - never defer to 'run this script'\n"
            f"  • Include all results, visualizations, and recommendations directly\n"
            f"  • User should have everything they need without additional steps\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"CONSTRAINTS: Keep total plan under 200 words. Use concise bullet points. No preamble or meta-commentary."
        )
        return generate_text(prompt, temperature=0.25, max_tokens=1200)

    def critique_plan_prompt(self, text, plan):
        prompt = (
            f"ROLE: You are a critical plan reviewer analyzing execution plans for flaws and inefficiencies.\n\n"
            f"TASK: Identify 3-6 concrete issues in this plan that would harm output quality or waste resources.\n\n"
            f"USER REQUEST:\n{text}\n\n"
            f"PLAN TO REVIEW:\n{plan}\n\n"
            f"FOCUS AREAS (prioritize in this order):\n"
            f"1. LOGICAL CONTRADICTIONS: Any self-contradictory statements, conflicting constraints, or dogmatic assertions?\n"
            f"2. CORE INSTRUCTION WEAKNESS: Is it vague, too broad, or misaligned with user needs?\n"
            f"3. INEFFICIENT STEPS: Redundant blocks, unnecessary complexity, or steps that don't advance the goal?\n"
            f"4. VAGUE RUBRIC: Are success criteria measurable and falsifiable, or just aspirational fluff?\n"
            f"5. TOKEN WASTE: Excessive verbosity, over-specification, or requesting more data than needed?\n"
            f"6. ACCURACY RISKS: Missing verification steps, hallucination risks, or weak fact-checking?\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return exactly 3-6 issues as a numbered list. For each issue:\n"
            f"- State the ISSUE (be specific about what's wrong)\n"
            f"- Explain the IMPACT (why this matters for quality/efficiency)\n"
            f"- Suggest a FIX (concrete action to resolve it)\n\n"
            f"Example format:\n"
            f"1. ISSUE: Core instruction says 'analyze data' without specifying what to analyze.\n"
            f"   IMPACT: Model may produce irrelevant or shallow analysis.\n"
            f"   FIX: Specify exact metrics or patterns to extract from the data.\n\n"
            f"Be ruthlessly critical but constructive. Focus on high-impact improvements."
        )
        return generate_text(prompt, temperature=0.2, max_tokens=400)

    def fixer_plan_prompt(self, text, plan, critique):
        prompt = (
            f"ROLE: You are refining an execution plan based on critical feedback.\n\n"
            f"TASK: Produce PLAN v2 that addresses ALL issues raised in the critique while maintaining the same structure.\n\n"
            f"USER REQUEST:\n{text}\n\n"
            f"ORIGINAL PLAN:\n{plan}\n\n"
            f"CRITIQUE (YOU MUST FIX THESE):\n{critique}\n\n"
            f"REQUIREMENTS FOR PLAN v2:\n"
            f"- Keep the EXACT same 7 sections: Core Instruction, Constraints, Personality, Execution Steps, Evidence To Extract, Quality Rubric, Output Schema\n"
            f"- Address EVERY issue mentioned in the critique explicitly\n"
            f"- Strengthen any vague or weak elements\n"
            f"- Add verification steps if missing\n"
            f"- Ensure visualizations (if relevant) will be produced as actual matplotlib plots saved to files\n"
            f"- Confirm final answer delivers complete results without deferring to external scripts\n"
            f"- Maintain conciseness (under 200 words total)\n\n"
            f"OUTPUT: Return ONLY the refined plan with the 7 numbered sections. No preamble, no explanations of changes."
        )
        return generate_text(prompt, temperature=0.25, max_tokens=600)


class UseCodeToolBlock(PipelineBlock):
    description = "Run Python code to compute or analyze data. Requires data.extract_info describing what to pull."
    
    def __init__(self, max_attempts: int = 8):
        self.max_attempts = max(1, max_attempts)

    def __call__(self, text, extract_info=None) -> dict:
        attempts_summary = []
        previous_error = None
        previous_code = None
        final_output = {}
        final_code = ""

        plot_requested = self._should_generate_plot(text, extract_info)

        for attempt in range(1, self.max_attempts + 1):
            raw_code = self.create_code_prompt(
                text,
                extract_info,
                previous_code=previous_code,
                previous_error=previous_error,
                attempt=attempt,
                plot_requested=plot_requested,
            )
            code = self._extract_code_from_response(raw_code)
            code, applied_fixes = self._ensure_required_imports(code)
            previous_code = code
            # Guard: code may be None; ensure it's a string before calling strip()
            if not (code and str(code).strip()):
                logger.warning("UseCodeToolBlock attempt %d returned empty code", attempt)
                attempts_summary.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "error": "Empty code returned",
                        "raw_code": raw_code,
                        "stdout": "",
                        "applied_fixes": [],
                    }
                )
                previous_error = "Model returned empty code."
                continue

            output = python_exec_tool(code, save_plots=plot_requested)
            final_output = output
            final_code = code

            success = output.get("success", False)
            stdout = output.get("output", "")
            error_msg = output.get("error")
            structured_output = self._parse_structured_output(stdout)

            attempts_summary.append(
                {
                    "attempt": attempt,
                    "success": success,
                    "error": error_msg,
                    "raw_code": raw_code,
                    "stdout": stdout,
                    "applied_fixes": applied_fixes,
                    "structured_output": structured_output,
                }
            )

            if success:
                validation_error = self._validate_structured_output(structured_output, plot_requested)
                if validation_error:
                    logger.warning(
                        "UseCodeToolBlock attempt %d produced invalid structured output: %s",
                        attempt,
                        validation_error,
                    )
                    success = False
                    output["success"] = False
                    output["error"] = validation_error
                    attempts_summary[-1]["success"] = False
                    attempts_summary[-1]["error"] = validation_error
                    previous_error = (
                        validation_error
                        + "\nGuidance: Print a single json.dumps payload with keys 'answer', 'details', and 'metadata'."
                    )
                    final_output = output
                    final_code = code
                    continue

                logger.info("UseCodeToolBlock succeeded on attempt %d", attempt)
                break

            logger.warning(
                "UseCodeToolBlock attempt %d failed: %s", attempt, (error_msg or stdout)[:200]
            )
            previous_error = (error_msg or stdout or "Execution failed without message.")
            
            # Add specific guidance for common errors
            if previous_error and "name 'np' is not defined" in previous_error:
                previous_error += \
                    "\nGuidance: Rewrite the solution without relying on numpy; use pure Python data structures."
            elif previous_error and ("googlesearch" in previous_error.lower() or "requests" in previous_error.lower() or "beautifulsoup" in previous_error.lower()):
                previous_error += \
                    "\nGuidance: Web scraping modules are NOT available. You already have the article content provided. Work ONLY with the data you have. Do NOT attempt web searches or HTTP requests."
            elif previous_error and "ModuleNotFoundError" in previous_error:
                previous_error += \
                    "\nGuidance: Only use standard library modules (json, re, datetime, etc.) and matplotlib. External packages are not available. Work with the data already provided to you."
            elif previous_error and "sympy" in previous_error.lower():
                previous_error += \
                    "\nGuidance: sympy is NOT available. Solve the system of equations manually using substitution/elimination method or matrix operations with pure Python."
            elif previous_error and ("list indices must be integers" in previous_error or "Symbol" in previous_error):
                previous_error += \
                    "\nGuidance: You're trying to use symbolic variables as list indices. Use numeric calculations only - solve equations algebraically by hand, then calculate the numeric result."

        output_output = final_output.get("output", "") if final_output else ""
        output_files = final_output.get("plots", []) if final_output else []
        structured_output = self._parse_structured_output(output_output)

        response = {
            "code_request": text,
            "extract_info": extract_info or "",
            "output": output_output,
            "files": output_files,
            "plots": output_files,
            "raw_result": final_output,
            "attempts": attempts_summary,
            "success": final_output.get("success", False) if final_output else False,
            "final_code": final_code,
            "structured_output": structured_output,
        }
        return response

    def _validate_structured_output(self, structured_output, plot_requested: bool) -> Optional[str]:
        if structured_output is None:
            return (
                "Code tool must emit a JSON object via json.dumps; no JSON payload was captured."
            )

        if not isinstance(structured_output, dict):
            return "Structured output must be a JSON object."

        required_keys = {"answer", "details", "metadata"}
        missing = sorted(required_keys - structured_output.keys())
        if missing:
            return (
                "Structured output missing required keys: " + ", ".join(missing)
            )

        metadata = structured_output.get("metadata")
        if not isinstance(metadata, dict):
            return "metadata field must be a JSON object."
        
        # CRITICAL: Check if the code reported an error in the output
        if metadata.get("error") is True:
            answer = structured_output.get("answer", "")
            return f"Code execution reported an error: {answer}"
        
        # Check if the answer contains error indicators
        answer = str(structured_output.get("answer", "")).lower()
        error_indicators = ["error occurred", "exception", "failed", "traceback"]
        if any(indicator in answer for indicator in error_indicators):
            return f"Code output contains error message: {structured_output.get('answer', '')}"

        if plot_requested and not metadata.get("plot_saved"):
            return (
                "metadata['plot_saved'] must be True when a plot is requested."
            )

        return None

    def create_code_prompt(
        self,
        text,
        extract_info=None,
        previous_code=None,
        previous_error=None,
        attempt: int = 1,
        plot_requested: bool = False,
    ):
        prompt = (
            f"ROLE: You are an expert Python programmer writing code to extract information and generate results.\n\n"
            f"OBJECTIVE: Write clean, efficient Python code to fulfill the request below.\n\n"
            f"USER REQUEST:\n{text}\n\n"
            f"INFORMATION TO EXTRACT:\n{extract_info or 'Extract relevant insights, calculations, or patterns from the provided data.'}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"CRITICAL CONSTRAINTS - READ CAREFULLY:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"DATA ACCESS:\n"
            f"  ✓ ALLOWED: Work with data already present in the prompt/context above\n"
            f"  ✓ ALLOWED: Standard library (json, re, datetime, math, collections, itertools, etc.)\n"
            f"  ✓ ALLOWED: matplotlib.pyplot for visualizations\n"
            f"  ✗ FORBIDDEN: Web scraping, HTTP requests, or external data fetching\n"
            f"  ✗ FORBIDDEN: Packages like requests, beautifulsoup, googlesearch, selenium, urllib, httplib\n"
            f"  ✗ FORBIDDEN: Database connections, file system operations beyond temp plots\n"
            f"  ✗ FORBIDDEN: Heavy dependencies like numpy, pandas, scipy (use standard library instead)\n\n"
            f"CODE QUALITY:\n"
            f"  • Write production-ready code with proper error handling\n"
            f"  • Use descriptive variable names and add brief inline comments for complex logic\n"
            f"  • Handle edge cases (empty data, division by zero, missing keys, etc.)\n"
            f"  • Prefer clarity over cleverness - code should be maintainable\n"
            f"  • Keep it efficient - avoid unnecessary loops or redundant calculations\n\n"
            f"OUTPUT REQUIREMENTS:\n"
            f"  • Code MUST print EXACTLY ONE JSON object at the end using json.dumps()\n"
            f"  • JSON must contain these exact keys:\n"
            f"    - 'answer': (string) Clear summary of findings in 2-4 sentences\n"
            f"    - 'details': (dict/list) Structured data like tables, metrics, intermediate values\n"
            f"    - 'metadata': (dict) Include flags like 'plot_saved', calculation parameters, data sources\n"
            f"  • Print NOTHING else to stdout - no debug messages, no extra text\n"
            f"  • The JSON output is consumed by downstream processing - it must be valid\n\n"
            f"DELIVERABLE COMPLETENESS:\n"
            f"  • Provide final analytical results DIRECTLY in the JSON\n"
            f"  • DO NOT generate 'example scripts' for users to run\n"
            f"  • DO NOT include instructions like 'save this and execute'\n"
            f"  • The output IS the final result, not a template\n\n"
        )

        if plot_requested:
            prompt += (
                f"═════════════════════════════════════════════════════════════════\n"
                f"🎨 VISUALIZATION REQUIRED:\n"
                f"═════════════════════════════════════════════════════════════════\n\n"
                f"You MUST generate a high-quality matplotlib plot. Follow this checklist:\n\n"
                f"SETUP:\n"
                f"  1. Import: import matplotlib.pyplot as plt\n"
                f"  2. Create figure: plt.figure(figsize=(10, 6)) or larger if needed\n\n"
                f"PLOTTING:\n"
                f"  3. Choose appropriate chart type:\n"
                f"     • Line plot: trends over time, continuous data (plt.plot)\n"
                f"     • Bar chart: comparisons, categories (plt.bar)\n"
                f"     • Scatter plot: correlations, distributions (plt.scatter)\n"
                f"     • Pie chart: proportions, percentages (plt.pie)\n"
                f"     • Histogram: frequency distributions (plt.hist)\n"
                f"  4. Use clear colors and markers for readability\n"
                f"  5. Add gridlines if helpful: plt.grid(True, alpha=0.3)\n\n"
                f"LABELS & ANNOTATIONS:\n"
                f"  6. X-axis label: plt.xlabel('Clear descriptive label', fontsize=12)\n"
                f"  7. Y-axis label: plt.ylabel('Clear descriptive label', fontsize=12)\n"
                f"  8. Title: plt.title('Descriptive Title That Explains The Visualization', fontsize=14, fontweight='bold')\n"
                f"  9. Legend if multiple series: plt.legend(loc='best')\n"
                f"  10. Consider adding data labels or annotations for key points\n\n"
                f"FINALIZATION:\n"
                f"  11. Apply layout: plt.tight_layout() (prevents label cutoff)\n"
                f"  12. Save: plt.show() - THIS IS CRITICAL\n"
                f"      • The system auto-saves the figure to a file\n"
                f"      • DO NOT use plt.savefig() yourself\n"
                f"      • Call plt.show() EXACTLY ONCE at the very end\n\n"
                f"JSON OUTPUT FOR PLOTS:\n"
                f"  • 'answer': Describe what the visualization shows and key insights\n"
                f"  • 'details': Include the raw data used (as dict/list) so users can verify\n"
                f"  • 'metadata': {{\n"
                f"      'plot_saved': True,\n"
                f"      'chart_type': 'line/bar/scatter/pie/histogram',\n"
                f"      'data_points': <number of points plotted>,\n"
                f"      'title': '<your plot title>'\n"
                f"    }}\n\n"
                f"The plot file path will be captured automatically and embedded in the final answer.\n\n"
            )

        if attempt > 1:
            prompt += (
                f"═════════════════════════════════════════════════════════════════\n"
                f"⚠️ PREVIOUS ATTEMPT FAILED (Attempt #{attempt}):\n"
                f"═════════════════════════════════════════════════════════════════\n\n"
                f"Carefully study the error below and correct your code.\n"
            )
        
        if previous_error:
            prompt += (
                f"\nERROR FROM PREVIOUS ATTEMPT:\n"
                f"────────────────────────────────────────────────────────────────\n"
                f"{previous_error}\n"
                f"────────────────────────────────────────────────────────────────\n\n"
                f"DEBUGGING CHECKLIST:\n"
                f"  • If NameError: Import the missing module or define the variable\n"
                f"  • If KeyError: Check that the key exists before accessing (use .get())\n"
                f"  • If IndexError: Verify list/array length before accessing indices\n"
                f"  • If TypeError: Check data types - convert strings to int/float as needed\n"
                f"  • If ZeroDivisionError: Add checks before division operations\n"
                f"  • If SyntaxError: Review brackets, quotes, indentation carefully\n\n"
            )
        
        if previous_code:
            prompt += (
                f"PREVIOUS CODE (for reference):\n"
                f"────────────────────────────────────────────────────────────────\n"
                f"{previous_code}\n"
                f"────────────────────────────────────────────────────────────────\n\n"
                f"Provide a COMPLETE corrected version (not a patch). Fix the root cause.\n\n"
            )
        
        prompt += (
            f"OUTPUT: Return ONLY the Python code. No explanations, no markdown formatting, no comments outside the code.\n"
            f"The code will be executed immediately in an isolated environment."
        )
        
        return generate_text(prompt, temperature=0.8)

    def _should_generate_plot(self, text: str, extract_info: str | None) -> bool:
        combined = " ".join(filter(None, [text, extract_info])).lower()
        keywords = ("plot", "graph", "chart", "visualize", "visualise", "diagram", "curve")
        return any(keyword in combined for keyword in keywords)

    def _parse_structured_output(self, stdout: str | None):
        if not stdout:
            return None

        cleaned = stdout.strip()
        if not cleaned:
            return None

        # Attempt direct JSON parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to locate the last JSON object in the text
        json_candidates = re.findall(r"\{.*\}", cleaned, re.DOTALL)
        for candidate in reversed(json_candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return None
    def _ensure_required_imports(self, code: str):
        # Defensive: accept None for code and treat as empty
        if not (code and str(code).strip()):
            return code or "", []

        imports_to_add = []

        if "json." in code and "import json" not in code:
            imports_to_add.append("import json")
        if "np." in code and "import numpy as np" not in code:
            imports_to_add.append("import numpy as np")
        if "plt." in code and "import matplotlib.pyplot as plt" not in code:
            imports_to_add.append("import matplotlib.pyplot as plt")

        if not imports_to_add:
            return code, []

        logger.info(
            "UseCodeToolBlock inserting missing imports: %s",
            ", ".join(imports_to_add),
        )
        applied = [f"auto_import:{imp}" for imp in imports_to_add]
        return "\n".join(imports_to_add + ["", code]), applied

    def _extract_code_from_response(self, response: str) -> str:
        # Defensive: response may be None
        trimmed = (response or "").strip()
        if not trimmed:
            return ""

        if trimmed.startswith("{"):
            try:
                payload = json.loads(trimmed)
                if isinstance(payload, dict) and "code" in payload:
                    return payload["code"]
            except json.JSONDecodeError:
                pass

        if "```" in trimmed:
            sections = trimmed.split("```")
            if len(sections) >= 3:
                # sections[1] may include language identifier
                candidate = sections[1]
                candidate = candidate or ""
                if candidate.strip().startswith("python"):
                    candidate = "\n".join(candidate.splitlines()[1:])
                return candidate.strip()

        return trimmed


class UseInternetToolBlock(PipelineBlock):
    description = "Search the web. Requires data.search_query and optional data.link_num (1-10). Gracefully handles fetch failures."
    
    def __call__(self, query, num_links):
        output = online_query_tool(query, num_links)
        summary = output.get("summary", "")
        articles = output.get("articles", [])
        fetch_status = output.get("fetch_status", "UNKNOWN")
        error = output.get("error", None)
        
        # Format the response based on fetch status
        if fetch_status == "SUCCESS":
            articles_formatted = "\n".join([f"- {a.get('title', 'Untitled')}: {a.get('link', 'No link')}" for a in articles if a])
            text = f"""Web search completed successfully for: "{query}"

Summary of findings:
{summary}

Sources:
{articles_formatted if articles_formatted else "(No article links available)"}
"""
        else:
            # Fetch failed - explicitly communicate this to prevent hallucination
            text = f"""⚠️ WEB SEARCH FAILED for: "{query}"

Reason: {error or 'Unknown error'}

Status: {fetch_status}

Details:
{summary}

IMPORTANT: Cannot provide information about this topic from web sources. 
Please use general knowledge or ask for alternative assistance.
"""
        
        return text


class MathImprovementBlock(PipelineBlock):
    description = "Validate and improve mathematical reasoning. Catches algebra/calculus/logic errors and provides corrections."
    
    def __call__(self, text, math_content, verify_with_code=True) -> dict:
        """
        Validate mathematical reasoning and provide corrections if needed.
        
        Args:
            text: The original query or context
            math_content: The mathematical reasoning or calculation to validate
            verify_with_code: Whether to attempt numerical verification via Python code
            
        Returns:
            dict with validation status, errors found, corrected version, and code verification
        """
        # First pass: LLM-based validation
        validation = self.validate_math_prompt(text, math_content)
        
        # Second pass: Self-verification through independent re-derivation
        if not validation.get("is_correct", False):
            logger.info("Math validation found errors. Attempting self-verification...")
            verification = self.self_verify_prompt(text, math_content, validation)
            validation["self_verification"] = verification
        
        # Third pass: Numerical verification via code (if applicable)
        code_verification = None
        if verify_with_code:
            code_verification = self.verify_with_computation(text, math_content, validation)
            validation["code_verification"] = code_verification
        
        return {
            "original_math": math_content,
            "validation": validation,
            "verified_with_code": code_verification is not None,
        }
    
    def validate_math_prompt(self, context, math_content):
        """Validate mathematical reasoning and identify errors."""
        prompt = (
            f"ROLE: You are an expert mathematics validator specializing in error detection and correction.\n\n"
            f"OBJECTIVE: Carefully validate the mathematical reasoning below. Identify ANY errors and provide corrections.\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"CONTEXT:\n{context}\n\n"
            f"MATHEMATICAL REASONING TO VALIDATE:\n{math_content}\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n\n"
            f"VALIDATION CHECKLIST:\n"
            f"1. ALGEBRAIC CORRECTNESS\n"
            f"   ✓ Are all equation manipulations valid?\n"
            f"   ✓ Are factorizations correct?\n"
            f"   ✓ Are simplifications accurate?\n"
            f"   ✓ Check for sign errors, distribution errors, exponent errors\n\n"
            f"2. CALCULUS ACCURACY (if applicable)\n"
            f"   ✓ Are derivatives computed correctly (chain rule, product rule, quotient rule)?\n"
            f"   ✓ Are integrals evaluated properly?\n"
            f"   ✓ Are limits handled correctly?\n"
            f"   ✓ Are boundary conditions applied?\n\n"
            f"3. LOGICAL CONSISTENCY\n"
            f"   ✓ Do the steps follow logically from one to the next?\n"
            f"   ✓ Are assumptions stated and justified?\n"
            f"   ✓ Are special cases (division by zero, undefined values) handled?\n\n"
            f"4. NUMERICAL ACCURACY\n"
            f"   ✓ Are numerical calculations correct?\n"
            f"   ✓ Are decimal conversions accurate?\n"
            f"   ✓ Are rounding errors considered?\n\n"
            f"5. DIMENSIONAL ANALYSIS (if applicable)\n"
            f"   ✓ Are units consistent throughout?\n"
            f"   ✓ Are unit conversions correct?\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"OUTPUT FORMAT - STRICT JSON REQUIRED:\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n\n"
            f"You MUST return ONLY a valid JSON object. No explanatory text before or after.\n"
            f"Do NOT wrap the JSON in markdown code blocks (no ```json or ```).\n"
            f"Return the raw JSON object directly.\n\n"
            f"Required structure:\n\n"
            f"{{\n"
            f"  \"is_correct\": true,\n"
            f"  \"errors_found\": [],\n"
            f"  \"severity\": \"none\",\n"
            f"  \"corrected_reasoning\": \"No corrections needed\",\n"
            f"  \"final_answer\": \"The reasoning is mathematically sound\",\n"
            f"  \"explanation\": \"Brief explanation of validation\"\n"
            f"}}\n\n"
            f"OR if errors found:\n\n"
            f"{{\n"
            f"  \"is_correct\": false,\n"
            f"  \"errors_found\": [\n"
            f"    {{\n"
            f"      \"error_type\": \"algebraic\",\n"
            f"      \"location\": \"Step 2: factorization\",\n"
            f"      \"description\": \"Incorrect application of difference of squares\",\n"
            f"      \"consequence\": \"Final answer is off by a factor of 2\"\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"severity\": \"major\",\n"
            f"  \"corrected_reasoning\": \"Step-by-step corrected version here\",\n"
            f"  \"final_answer\": \"Correct final answer\",\n"
            f"  \"explanation\": \"What was wrong and how it was fixed\"\n"
            f"}}\n\n"
            f"CRITICAL RULES:\n"
            f"  ✓ ONLY report errors you can mathematically verify\n"
            f"  ✓ NEVER hallucinate errors or corrections not supported by math\n"
            f"  ✓ If you're unsure, mark severity as 'minor' and explain uncertainty\n"
            f"  ✓ Show all intermediate steps in corrected_reasoning\n"
            f"  ✗ DO NOT guess at complex proofs - only validate concrete calculations\n"
            f"  ✗ DO NOT invent mathematical concepts to justify corrections\n\n"
            f"Return ONLY the JSON object, no other text."
        )
        
        raw_response = generate_text(prompt, temperature=0.3, max_tokens=2000)
        
        # Try to parse JSON response with enhanced error handling
        try:
            cleaned = raw_response.strip()
            
            # Strip markdown code blocks if present (like router does)
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]  # Remove opening ```json or ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove closing ```
                cleaned = '\n'.join(lines).strip()
            
            # Try direct JSON parse first
            try:
                validation_result = json.loads(cleaned)
                logger.info(f"Math validation complete. Correct: {validation_result.get('is_correct')}")
                return validation_result
            except json.JSONDecodeError:
                # Fallback: Extract JSON object from response
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    validation_result = json.loads(match.group(0))
                    logger.info(f"Math validation complete (extracted). Correct: {validation_result.get('is_correct')}")
                    return validation_result
                else:
                    # Log the actual response for debugging
                    logger.warning(f"Could not extract JSON from math validation response. Response preview: {raw_response[:500]}")
                    return {
                        "is_correct": False,
                        "errors_found": [],
                        "severity": "unknown",
                        "corrected_reasoning": "Could not parse validation response - no JSON object found",
                        "final_answer": "VALIDATION FAILED",
                        "explanation": f"Math validation returned unparseable response. Preview: {raw_response[:200]}"
                    }
        except Exception as e:
            logger.error(f"Math validation parsing error: {e}. Response preview: {raw_response[:300]}")
            return {
                "is_correct": False,
                "errors_found": [],
                "severity": "unknown",
                "corrected_reasoning": f"Validation error: {str(e)}",
                "final_answer": "VALIDATION FAILED",
                "explanation": f"Math validation encountered error: {str(e)}"
            }
    
    def self_verify_prompt(self, context, math_content, validation_result):
        """
        Independent re-derivation to verify the correction is actually correct.
        This acts as a second opinion - solving the problem from scratch AND checking by substitution.
        """
        prompt = (
            f"ROLE: You are a second mathematics expert providing independent verification.\n\n"
            f"OBJECTIVE: Verify the solution by BOTH solving independently AND checking by substitution.\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"CONTEXT:\n{context}\n\n"
            f"PREVIOUS VALIDATION FOUND THESE ISSUES:\n"
            f"{json.dumps(validation_result.get('errors_found', []), indent=2)}\n\n"
            f"CORRECTED ANSWER CLAIMED:\n{validation_result.get('final_answer', 'N/A')}\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n\n"
            f"YOUR VERIFICATION STEPS:\n"
            f"1. **Solve independently**: Work through the problem from first principles\n"
            f"2. **Check by substitution**: Take the claimed answer and substitute it back into the original problem\n"
            f"3. **Verify consistency**: Do both methods agree?\n\n"
            f"EXAMPLE FOR SYSTEM OF EQUATIONS:\n"
            f"  - If solution claims a_r=10, d_r=-5, a_c=20, d_c=8\n"
            f"  - Substitute these values back into ALL original equations\n"
            f"  - Check: Does a_r + 4*d_r + a_c + 4*d_c = 0? → 10 + 4*(-5) + 20 + 4*8 = 10 - 20 + 20 + 32 = 42 ≠ 0 (ERROR!)\n"
            f"  - This proves the solution is wrong\n\n"
            f"OUTPUT FORMAT (JSON only):\n"
            f"{{\n"
            f"  \"independent_solution\": \"Your step-by-step solution\",\n"
            f"  \"independent_answer\": \"Your final answer\",\n"
            f"  \"substitution_check\": \"Results of substituting claimed answer back into problem\",\n"
            f"  \"substitution_passes\": true/false,\n"
            f"  \"matches_correction\": true/false,\n"
            f"  \"confidence\": \"high/medium/low\",\n"
            f"  \"notes\": \"Any discrepancies or concerns\"\n"
            f"}}\n\n"
            f"CRITICAL: The substitution_check is the most important validation. If the answer doesn't satisfy\n"
            f"the original problem when substituted back, it's WRONG regardless of how elegant the derivation is.\n\n"
            f"Return ONLY the JSON object."
        )
        
        raw_response = generate_text(prompt, temperature=0.2, max_tokens=1500)
        
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines).strip()
            
            return json.loads(cleaned)
        except:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"error": "Could not parse self-verification response"}
    
    def verify_with_computation(self, context, math_content, validation_result):
        """
        Attempt to verify the mathematical result using Python computation.
        This provides numerical ground truth when applicable.
        """
        prompt = (
            f"ROLE: You are writing Python code to numerically verify a mathematical result.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"MATHEMATICAL CLAIM TO VERIFY:\n{validation_result.get('final_answer', math_content)}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"TASK: Write Python code that computes the result numerically to verify correctness.\n\n"
            f"CONSTRAINTS:\n"
            f"  • Use only standard library (math, cmath, fractions, decimal, etc.)\n"
            f"  • DO NOT use numpy, scipy, sympy (not available)\n"
            f"  • Print a SINGLE JSON object with:\n"
            f"    - 'computed_value': the numerical result\n"
            f"    - 'verification_status': 'VERIFIED', 'FAILED', or 'NOT_APPLICABLE'\n"
            f"    - 'details': explanation of what was computed\n\n"
            f"If the problem is purely symbolic (no numerical answer possible), return:\n"
            f"{{\n"
            f"  \"computed_value\": null,\n"
            f"  \"verification_status\": \"NOT_APPLICABLE\",\n"
            f"  \"details\": \"Problem is symbolic/algebraic - numerical verification not applicable\"\n"
            f"}}\n\n"
            f"RETURN ONLY THE PYTHON CODE (no markdown, no explanations)."
        )
        
        raw_code = generate_text(prompt, temperature=0.3, max_tokens=800)
        
        # Extract code if wrapped in markdown
        code = raw_code.strip()
        if "```" in code:
            sections = code.split("```")
            if len(sections) >= 3:
                candidate = sections[1]
                if candidate.strip().startswith("python"):
                    candidate = "\n".join(candidate.splitlines()[1:])
                code = candidate.strip()
        
        # Execute the verification code
        try:
            result = python_exec_tool(code, save_plots=False)
            if result.get("success"):
                stdout = result.get("output", "")
                # Try to parse JSON from stdout
                try:
                    verification_data = json.loads(stdout.strip())
                    logger.info(f"Code verification status: {verification_data.get('verification_status')}")
                    return verification_data
                except:
                    # Try to extract JSON from output
                    match = re.search(r'\{.*\}', stdout, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
            return {
                "computed_value": None,
                "verification_status": "FAILED",
                "details": f"Code execution failed: {result.get('error', 'Unknown error')}"
            }
        except Exception as e:
            logger.warning(f"Math code verification error: {e}")
            return {
                "computed_value": None,
                "verification_status": "ERROR",
                "details": f"Verification error: {str(e)}"
            }


class CreativeIdeaGeneratorBlockTool(PipelineBlock):
    description = "Brainstorm ideas. Requires data.criteria and optional data.choose_best (boolean)."
    
    def __call__(self, text, idea_criteria, choose_best=False) -> dict:
        ideas = self.create_idea_prompt(text, idea_criteria)
        response = {
            "ideas": ideas,
        }
        if choose_best:
            best = self.choose_best_idea_prompt(text, ideas)
            response["best_idea"] = best
        return response

    def create_idea_prompt(self, text, idea_criteria):
        prompt = (
            f"ROLE: You are an elite creative consultant specializing in breakthrough thinking across domains:\n"
            f"arts, technology, business strategy, scientific research, design, and social innovation.\n\n"
            f"OBJECTIVE: Generate EXACTLY 10 complete, ORIGINAL ideas that fulfill the user's request.\n"
            f"These must be genuine concepts, not variations of a single theme.\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"QUALITY STANDARDS - NON-NEGOTIABLE:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"ORIGINALITY:\n"
            f"  • Each idea must be DISTINCT - avoid presenting 10 variations of the same concept\n"
            f"  • Push beyond the obvious - reject your first impulse and think deeper\n"
            f"  • Blend unexpected elements from different domains (cross-pollination)\n"
            f"  • Challenge assumptions - what if the opposite were true?\n"
            f"  • Look for white space - what hasn't been done yet?\n\n"
            f"DEPTH:\n"
            f"  • Ideas should be specific enough to visualize and implement\n"
            f"  • Avoid generic concepts - make them tangible and concrete\n"
            f"  • Include unique mechanisms, angles, or approaches\n"
            f"  • Show how the idea differs from conventional wisdom\n\n"
            f"FRESHNESS:\n"
            f"  • Ban clichés, tired tropes, and overused frameworks\n"
            f"  • Seek surprising combinations and novel perspectives\n"
            f"  • Consider emerging trends, technologies, or cultural shifts\n"
            f"  • What would make someone say 'I've never thought of it that way'?\n\n"
            f"DIVERSITY:\n"
            f"  • Vary your approach - some radical, some practical, some experimental\n"
            f"  • Mix different scales - micro solutions and macro visions\n"
            f"  • Include both incremental innovations and paradigm shifts\n"
            f"  • Represent different risk-reward profiles\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"USER REQUEST:\n{text}\n\n"
            f"CRITERIA TO FULFILL:\n{idea_criteria}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"OUTPUT FORMAT - STRICT JSON REQUIRED:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"Return a JSON array of EXACTLY 10 objects. Each object must have:\n\n"
            f"{{\n"
            f"  \"title\": \"<Compelling 3-8 word concept title>\",\n"
            f"  \"justification\": \"<2-3 sentence explanation covering: what it is, why it's innovative, how it fulfills criteria>\"\n"
            f"}}\n\n"
            f"CRITICAL OUTPUT RULES:\n"
            f"  ✓ Return ONLY valid JSON - no prose before or after the array\n"
            f"  ✓ Each title should be distinctive and memorable\n"
            f"  ✓ Each justification should sell the idea's uniqueness\n"
            f"  ✗ DO NOT echo the criteria back as an 'idea'\n"
            f"  ✗ DO NOT restate the user's request verbatim\n"
            f"  ✗ DO NOT include explanatory text outside the JSON\n"
            f"  ✗ DO NOT use placeholder language like '[idea X]' or 'Option 1'\n\n"
            f"EXAMPLES OF WHAT TO AVOID:\n"
            f"  ❌ Title: 'Fulfill the criteria by doing X' (this echoes the criteria)\n"
            f"  ❌ Title: 'Idea 1', 'Concept A' (generic labels)\n"
            f"  ❌ Justification: 'This meets all the requirements' (empty statement)\n"
            f"  ❌ Including prose like: 'Here are 10 ideas that fulfill...'\n\n"
            f"EXAMPLES OF GOOD IDEAS:\n"
            f"  ✅ Title: 'Quantum Memory Palace for Language Learning'\n"
            f"     Justification: 'Combines spaced repetition with VR environments where vocabulary\n"
            f"                    materializes in contextually relevant 3D spaces, leveraging spatial\n"
            f"                    memory for 3x faster retention compared to traditional flashcards.'\n"
            f"  ✅ Title: 'Reverse Mentorship Pods for Corporate Innovation'\n"
            f"     Justification: 'Junior employees lead monthly strategy sessions teaching executives\n"
            f"                    about emerging tech and culture, breaking hierarchical thinking and\n"
            f"                    injecting fresh perspectives into stagnant decision-making.'\n\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"Think boldly. Surprise us. Be specific. Push boundaries.\n\n"
            f"Return your JSON array now:"
        )

        raw = generate_text(prompt, temperature=1.2, max_tokens=2000)

        # Try to parse JSON array from the raw output
        ideas_list = []
        try:
            import json as _json, re as _re
            m = _re.search(r"\[.*\]", raw, _re.DOTALL)
            if m:
                parsed = _json.loads(m.group(0))
                # If parsed is list of objects, convert to string summaries
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            title = item.get("title") or item.get("concept") or ""
                            justification = item.get("justification") or item.get("reason") or ""
                            ideas_list.append(f"{title.strip()} — {justification.strip()}")
                        else:
                            ideas_list.append(str(item).strip())
        except Exception:
            ideas_list = []

        # Fallback: parse numbered list if JSON parse failed
        if not ideas_list:
            import re as _re
            lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
            # Collect lines that look like numbered ideas (start with 1. 2. etc.)
            numbered = []
            for ln in lines:
                if _re.match(r"^\d+\.", ln):
                    numbered.append(_re.sub(r"^\d+\.\s*", "", ln))
            if numbered:
                ideas_list = numbered
            else:
                # Last resort: split by blank lines into idea chunks
                chunks = (raw or "").split("\n\n")
                ideas_list = [c.replace('\n', ' ').strip() for c in chunks if c.strip()][:10]

        # Sanitize/filter: remove items that merely echo the criteria or the prompt
        def _normalize(s: str) -> str:
            return _re.sub(r"\s+", " ", (s or "").strip()).lower()

        norm_criteria = _normalize(idea_criteria)
        norm_prompt = _normalize(text)
        filtered = []
        for idea in ideas_list:
            n = _normalize(idea)
            if not n:
                continue
            # Skip if idea is identical to criteria or prompt
            if norm_criteria and (norm_criteria in n or n in norm_criteria):
                logger.warning("Dropping idea that repeats criteria: %s", idea)
                continue
            if norm_prompt and (norm_prompt in n or n in norm_prompt):
                logger.warning("Dropping idea that repeats prompt: %s", idea)
                continue
            filtered.append(idea)

        # Ensure we return at most 10 items
        return filtered[:10]
    
    def choose_best_idea_prompt(self, text, ideas, choose_best=True):
        from src import utility

        # Ensure ideas is a list of strings
        if isinstance(ideas, str):
            # Try to split by numbered list
            idea_lines = [line.strip() for line in ideas.splitlines() if line.strip()]
            ideas_list = idea_lines
        elif isinstance(ideas, list):
            ideas_list = ideas
        else:
            ideas_list = [str(ideas)]

        criteria = (
            "Alignment with user's request; creativity and originality; feasibility and impact; "
            "absence of dogma; adherence to constraints; uniqueness among alternatives."
        )

        try:
            eval_result = utility.evaluate_ideas_via_client(text, ideas_list, criteria)
        except Exception as e:
            # Fallback to simple deterministic selection (first idea)
            logger.warning('Idea evaluation failed, falling back: %s', str(e))
            selected_index = 0
            feedback = ['Evaluation failed - fallback selection'] * len(ideas_list)
        else:
            selected_index = int(eval_result.get('selected_index', 0) or 0)
            feedback = eval_result.get('feedback', [])

        # Clamp index
        if selected_index < 0 or selected_index >= len(ideas_list):
            selected_index = 0

        selected_idea = ideas_list[selected_index]

        # Expand the chosen idea into a full, detailed deliverable
        expansion_prompt = (
            f"ROLE: You are transforming a selected creative concept into a complete, high-quality deliverable.\n\n"
            f"SELECTED IDEA (the winner from evaluation):\n{selected_idea}\n\n"
            f"ORIGINAL USER REQUEST (for context):\n{text}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"TASK: Expand this idea into a FULL, POLISHED final output.\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"EXPANSION GUIDELINES:\n\n"
            f"IF CREATIVE WORK (poem, story, artwork description, design):\n"
            f"  • Produce the COMPLETE creative piece, not just an outline\n"
            f"  • Honor all constraints (length, style, tone, theme)\n"
            f"  • Aim for publication/gallery quality - make it memorable\n"
            f"  • Use vivid sensory details, precise language, emotional depth\n"
            f"  • Avoid clichés and generic imagery\n"
            f"  • Let the idea shine through with originality and craft\n\n"
            f"IF PRACTICAL/STRATEGIC WORK (plan, strategy, solution):\n"
            f"  • Provide a detailed implementation plan with concrete steps\n"
            f"  • Include examples, frameworks, or methodologies\n"
            f"  • Address potential challenges and mitigation strategies\n"
            f"  • Specify expected outcomes and success metrics\n"
            f"  • Make it actionable - readers should know exactly what to do\n\n"
            f"RATIONALE REQUIREMENT:\n"
            f"  • Include a brief 2-3 sentence rationale explaining:\n"
            f"    - Why this idea best fits the user's needs\n"
            f"    - What makes it superior to alternatives\n"
            f"    - How it uniquely addresses the request\n\n"
            f"QUALITY EXPECTATIONS:\n"
            f"  • Depth over breadth - go deep on this ONE idea\n"
            f"  • Professional quality - suitable for real-world use\n"
            f"  • Complete and self-contained - no placeholders or TODOs\n"
            f"  • Directly addresses the original request\n\n"
            f"OUTPUT FORMAT:\n"
            f"  • For creative work: The finished piece + rationale\n"
            f"  • For practical work: Detailed plan with steps/examples + rationale\n"
            f"  • Use markdown formatting for readability\n"
            f"  • Structure with clear headers and sections\n\n"
            f"Deliver the expanded concept now:"
        )

        expanded = generate_text(expansion_prompt, temperature=0.9, max_tokens=1000)

        return {
            "best_index": selected_index,
            "best_idea": selected_idea,
            "feedback": feedback,
            "expanded": expanded,
        }


class SynthesizeFinalAnswerBlock(PipelineBlock):
    description = "Produce the final response using all collected context. This block MUST be last."
    
    def __call__(self, prompt, plan, collected_info) -> dict:
        # CRITICAL: First check if web search failed - if so, REFUSE to generate content
        web_fetch_failed, failure_details = self._check_web_fetch_failure(collected_info)
        
        if web_fetch_failed:
            # If web search was required and failed, return ONLY the failure message
            # Do NOT try to synthesize or hallucinate content
            failure_response = (
                f"I'm sorry, but I was unable to search the web for your query.\n\n"
                f"**Error Details:**\n{failure_details}\n\n"
                f"**What happened:**\n"
                f"The web search returned no results. This could be due to:\n"
                f"• Network connectivity issues\n"
                f"• The search service being temporarily unavailable\n"
                f"• Your query being blocked or rate-limited\n"
                f"• Google News not having indexed content for this search\n\n"
                f"**What to try:**\n"
                f"• Rephrase your search query with different keywords\n"
                f"• Try a more general search term\n"
                f"• Wait a moment and try again\n"
                f"• If searching for very recent news, try again in a few minutes\n\n"
                f"I cannot provide information about this topic because the web search failed. "
                f"I will not fabricate or guess at news content, sources, or links."
            )
            
            return {
                "initial_response": failure_response,
                "score": {"clarity": 10, "logic": 10, "completeness": 8, "accuracy": 10},
                "final_response": failure_response,
            }
        
        # Normal flow: web search succeeded or wasn't needed
        initial_response = self.create_synthesis_prompt(prompt, plan, collected_info)
        scorer = self.scorer_prompt(prompt, initial_response)
        improved = self.improver_prompt(prompt, initial_response, plan, scorer)
        return {
            "initial_response": initial_response,
            "score": scorer,
            "final_response": improved,
        }
    
    def _check_web_fetch_failure(self, collected_info: dict) -> tuple:
        """Check if any web fetch tool returned a failure status. Returns (failed, details)."""
        for key, value in collected_info.items():
            if "use_internet_tool" in key and isinstance(value, str):
                if "WEB SEARCH FAILED" in value or "WEB FETCH FAILED" in value or "EXTRACTION FAILED" in value:
                    return True, value[:800]  # Return failure message
        return False, ""

    def create_synthesis_prompt(self, prompt, plan, collected_info:dict):
        collated, assets = self._collate_collected_info(collected_info)
        
        collected_info_text = json.dumps(collated, indent=2, ensure_ascii=False)
        
        # CRITICAL: Limit context size to prevent token quota exhaustion
        # Reduced from 50K to 30K chars to be more conservative
        MAX_CONTEXT_CHARS = 30000  # ~7.5K tokens, leaves room for plan, prompt, instructions
        if len(collected_info_text) > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Collected info is {len(collected_info_text):,} chars "
                f"(~{len(collected_info_text)//4:,} tokens). Truncating to {MAX_CONTEXT_CHARS:,} chars "
                f"to prevent quota exhaustion."
            )
            collected_info_text = collected_info_text[:MAX_CONTEXT_CHARS] + "\n... [TRUNCATED DUE TO SIZE]"
        
        assets_text = "\n".join(assets) if assets else "None"
        
        # Extract structured output from code tool if available
        code_tool_data = None
        for key, value in collected_info.items():
            if "use_code_tool" in key and isinstance(value, dict):
                code_tool_data = value.get("structured_output")
                break
        
        prompt_text = (
            f"You are synthesizing a final answer. PRIORITIZE the USER'S ORIGINAL PROMPT first and use the INTERNAL PLAN only as secondary execution guidance.\n"
            f"Follow the Output Schema exactly and ensure all Rubric criteria are met.\n\n"
        )
        
        prompt_text += (
            f"PROMPT (PRIMARY - use this to determine purpose and scope):\n{prompt}\n\n"
            f"PLAN (SECONDARY - use to shape steps but do NOT override user's explicit intent):\n{plan}\n\n"
            f"COLLECTED INFORMATION:\n{collected_info_text}\n\n"
        )
        
        if code_tool_data:
            prompt_text += f"CODE TOOL STRUCTURED OUTPUT:\n{json.dumps(code_tool_data, indent=2)}\n\n"
        
        # Add explicit date context
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Check if we actually have plots
        has_plots = assets and assets != ["None"] and len(assets) > 0
        
        prompt_text += f"IMPORTANT: Today's date is {current_date}. Use this date for all time-sensitive references.\n\n"
        
        if has_plots:
            prompt_text += f"AVAILABLE PLOTS (local file paths):\n{assets_text}\n\n"
        
        prompt_text += (
            f"CRITICAL REQUIREMENTS:\n"
            f"1. Return your answer in MARKDOWN format, NOT JSON. Write in clear, readable paragraphs.\n"
            f"2. If the Output Schema specifies JSON structure, use that as a GUIDE for organizing your content, but write it as formatted text with headers, lists, and paragraphs.\n"
            f"3. FORMATTING: For mathematical notation:\n"
            f"   - Use HTML tags like <sup> and <sub> for superscripts/subscripts (they will render properly)\n"
            f"   - For example: x<sup>2</sup>, x<sup>3</sup>, H<sub>2</sub>O\n"
            f"   - These will display correctly in the final output\n"
        )
        
        if has_plots:
            prompt_text += (
                f"4. PLOTS ARE AVAILABLE: Integrate EVERY plot naturally into your response using markdown: ![Description](file_path)\n"
                f"   - CRITICAL: ONLY use the EXACT file paths listed above - DO NOT invent or modify paths\n"
                f"   - You have {len(assets)} plot(s) available. Use all of them.\n"
                f"   - Embed images inline where they best support your explanation - do NOT create a separate 'Visuals' or 'Results' section\n"
                f"   - Each graph should appear near the text that discusses it\n"
                f"   - DO NOT mention plots if you cannot embed them with real paths\n"
                f"   - DO NOT create image references for plots 2, 3, etc. if only 1 plot was provided\n"
            )
        else:
            prompt_text += (
                f"4. NO PLOTS AVAILABLE: Do NOT create a 'Visuals' section. Do NOT mention charts, graphs, or images.\n"
                f"   - Do NOT use placeholder image paths like 'placeholder_*.png'\n"
                f"   - Do NOT create markdown image references like ![...](path) - you have ZERO plots\n"
                f"   - If visualizations would be helpful, mention this as a limitation, not as an existing feature\n"
            )
        
        prompt_text += (
            f"5. Use data from 'answer' and 'details' fields in the code tool output to write a comprehensive response\n"
            f"6. Ensure the response directly solves the user's request without asking them to run additional code\n"
            f"7. For any web links, place them in a 'References' section at the end\n"
            f"8. Include dates ONLY when they are critical to the content - avoid adding dates just as timestamps\n"
            f"9. CRITICAL - ANTI-HALLUCINATION RULE:\n"
            f"   - Do NOT fabricate data, statistics, quotes, or sources\n"
            f"   - Do NOT invent article titles, news summaries, or current events if web search failed\n"
            f"   - Do NOT create fake charts, graphs, or images\n"
            f"   - ONLY use information from the 'COLLECTED INFORMATION' section above\n"
            f"   - If web search FAILED (marked in collected info), say so explicitly\n"
            f"   - When uncertain about facts: state 'I cannot verify this' rather than guessing\n"
            f"10. CRITICAL: Deliver the answer DIRECTLY without meta-commentary\n"
            f"   - DO NOT explain your approach ('This response aims to...', 'The following will...', 'I've structured this...')\n"
            f"   - DO NOT reference the scoring/improvement process\n"
            f"   - DO NOT add concluding statements like 'This expanded response...'\n"
            f"   - Just provide the actual answer the user wants to read\n\n"
            f"Return only the final answer in markdown format. DO NOT wrap it in JSON."
        )
        # Increased temperature from 0.4 to 0.6 for more natural, less robotic responses
        return generate_text(prompt_text, temperature=0.6)
    
    def scorer_prompt(self, text, response):
        prompt = (
            f"ROLE: You are a rigorous Quality Assurance evaluator scoring AI-generated responses.\n\n"
            f"TASK: Evaluate this response across three dimensions and return a JSON score.\n\n"
            f"USER'S ORIGINAL REQUEST:\n{text}\n\n"
            f"RESPONSE TO EVALUATE:\n{response}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"SCORING DIMENSIONS (each 0-10):\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"1. CLARITY (0-10)\n"
            f"   How easy is this response to understand?\n"
            f"   • 9-10: Crystal clear, well-structured, perfect flow\n"
            f"   • 7-8: Clear and understandable with minor ambiguities\n"
            f"   • 5-6: Somewhat unclear, requires re-reading\n"
            f"   • 3-4: Confusing structure or language\n"
            f"   • 0-2: Nearly incomprehensible\n\n"
            f"   CHECK FOR:\n"
            f"   ✓ Logical organization with clear sections\n"
            f"   ✓ Smooth transitions between ideas\n"
            f"   ✓ Plain language appropriate to audience\n"
            f"   ✓ Proper formatting (headers, lists, emphasis)\n"
            f"   ✗ Jargon without explanation\n"
            f"   ✗ Run-on paragraphs or wall-of-text\n"
            f"   ✗ Unclear pronoun references\n\n"
            f"2. LOGIC (0-10)\n"
            f"   Is this response internally consistent and factually sound?\n"
            f"   • 9-10: Perfectly consistent, all claims verifiable\n"
            f"   • 7-8: Mostly sound with minor inconsistencies\n"
            f"   • 5-6: Some logical issues or unsupported claims\n"
            f"   • 3-4: Multiple contradictions or errors\n"
            f"   • 0-2: Fundamentally flawed or incoherent\n\n"
            f"   INSTANT ZERO OR NEAR-ZERO SCORES FOR:\n"
            f"   ⚠️ Placeholder images (placeholder_*.png) that don't exist\n"
            f"   ⚠️ Claiming 'see visualization below' when no image exists\n"
            f"   ⚠️ Multiple image references (plot_1, plot_2, plot_3) when only 1 exists\n"
            f"   ⚠️ Date inconsistencies or temporal impossibilities\n"
            f"   ⚠️ Self-contradictory statements within the text\n"
            f"   ⚠️ Hallucinated data, statistics, or sources\n"
            f"   ⚠️ Broken internal references ('see Section 5' but no Section 5)\n\n"
            f"   CHECK FOR:\n"
            f"   ✓ All image references point to real paths (/var/folders/ or /tmp/)\n"
            f"   ✓ Claims match provided data\n"
            f"   ✓ No contradictions between sections\n"
            f"   ✓ Dates and timeframes are accurate\n"
            f"   ✓ Cause-effect relationships make sense\n\n"
            f"3. ACTIONABILITY (0-10)\n"
            f"   Can the user immediately use or act on this response?\n"
            f"   • 9-10: Complete, specific, ready to implement immediately\n"
            f"   • 7-8: Mostly actionable with minor gaps\n"
            f"   • 5-6: Provides direction but lacks specifics\n"
            f"   • 3-4: Too abstract or theoretical to act on\n"
            f"   • 0-2: No practical value, purely conceptual\n\n"
            f"   CONTEXT MATTERS:\n"
            f"   • If user wanted poetry/art → abstract is GOOD (7-10)\n"
            f"   • If user wanted analysis/plan → abstract is BAD (0-3)\n"
            f"   • If user wanted data → must include actual data (not 'data shows X')\n"
            f"   • If user wanted steps → must have numbered, specific steps\n\n"
            f"   CHECK FOR:\n"
            f"   ✓ Concrete examples, not just principles\n"
            f"   ✓ Specific numbers, names, steps (not 'several' or 'many')\n"
            f"   ✓ Clear next steps or deliverables\n"
            f"   ✓ Answers the actual question asked\n"
            f"   ✗ Vague qualifiers ('often', 'typically', 'might')\n"
            f"   ✗ Generic advice that applies to everything\n"
            f"   ✗ Deferring to 'consult an expert' or 'do more research'\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"FEEDBACK REQUIREMENTS:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"Your feedback must be:\n"
            f"  • SPECIFIC: Name exact problems with line/section references\n"
            f"  • ACTIONABLE: Suggest concrete fixes, not vague improvements\n"
            f"  • PRIORITIZED: Start with most critical issues\n\n"
            f"Examples of GOOD feedback:\n"
            f"  ✅ 'Remove the image reference on line 12 (placeholder_chart.png) - this file doesn't exist.'\n"
            f"  ✅ 'Section 2 claims X but Section 4 contradicts with Y. Resolve by...'\n"
            f"  ✅ 'Replace vague statement \"many benefits\" with 3 specific benefits with examples.'\n\n"
            f"Examples of BAD feedback:\n"
            f"  ❌ 'Not actionable enough' (what specifically is missing?)\n"
            f"  ❌ 'Could be clearer' (which part? how?)\n"
            f"  ❌ 'Good but needs improvement' (empty statement)\n\n"
            f"IF ANY SCORE < 7: Your feedback MUST explain exactly what needs fixing.\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"OUTPUT FORMAT - EXACT JSON REQUIRED:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"Return ONLY this JSON structure (no other text):\n\n"
            f"{{\n"
            f"  \"clarity\": <0-10>,\n"
            f"  \"logic\": <0-10>,\n"
            f"  \"actionability\": <0-10>,\n"
            f"  \"feedback\": \"<Specific, actionable feedback with examples>\"\n"
            f"}}\n\n"
            f"Be tough but fair. High standards produce high quality."
        )
        return generate_text(prompt, temperature=0.1, max_tokens=400)

    def improver_prompt(self, text, response, plan, scorer):
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = (
            f"Today's date: {current_date}. Ensure all dates are accurate and only included when contextually relevant.\n\n"
            f"ROLE: You are a Quality Control specialist refining AI responses based on critical evaluation feedback.\n\n"
            f"OBJECTIVE: Produce an IMPROVED version of the response that addresses ALL issues identified in the scorer feedback.\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"USER'S ORIGINAL REQUEST:\n{text}\n\n"
            f"EXECUTION PLAN (for reference):\n{plan}\n\n"
            f"CURRENT RESPONSE (needs improvement):\n{response}\n\n"
            f"SCORER FEEDBACK (YOU MUST FIX EVERYTHING MENTIONED):\n{scorer}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"IMPROVEMENT METHODOLOGY:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"1. INTERPRET SCORES AS SEVERITY SIGNALS:\n"
            f"   • Score 0-4: CRITICAL FAILURE - requires complete rewrite\n"
            f"   • Score 5-6: MAJOR ISSUES - substantial revision needed\n"
            f"   • Score 7-8: MINOR FIXES - targeted improvements\n"
            f"   • Score 9-10: POLISH - small refinements\n\n"
            f"2. PRIORITIZE FIXES BY IMPACT:\n"
            f"   FIRST: Logic errors (contradictions, hallucinations, fake images)\n"
            f"   SECOND: Missing actionability (add specifics, examples, steps)\n"
            f"   THIRD: Clarity issues (restructure, simplify language)\n\n"
            f"3. ADDRESS SPECIFIC FEEDBACK POINTS:\n"
            f"   • Read each feedback item carefully\n"
            f"   • Make the EXACT changes suggested\n"
            f"   • Don't just rephrase - actually fix the underlying problem\n"
            f"   • If feedback says 'add X', you must add X\n"
            f"   • If feedback says 'remove Y', you must remove Y\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"CRITICAL IMAGE/VISUAL VALIDATION:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"MANDATORY CHECKS (skip these = automatic failure):\n"
            f"  ⚠️ Count ALL markdown image references: ![description](path)\n"
            f"  ⚠️ For EACH image reference, verify the path is real:\n"
            f"     • Real paths start with /var/folders/ or /tmp/\n"
            f"     • Fake paths: placeholder_*.png, plot_2.png (when only 1 exists), output.png\n"
            f"  ⚠️ If current response has 3 image refs but only 1 real plot exists:\n"
            f"     → DELETE the 2 fake image references completely\n"
            f"  ⚠️ If current response has a 'Visuals' or 'Results' section with fake images:\n"
            f"     → Either integrate the ONE real image inline OR delete the section entirely\n"
            f"  ⚠️ NEVER create new image references - only use what's explicitly provided\n\n"
            f"EMBEDDING GUIDELINES (for real images only):\n"
            f"  • Place images INLINE near the text they illustrate\n"
            f"  • Do NOT create separate 'Visualizations' sections\n"
            f"  • Format: ![Brief description](exact_path_provided)\n"
            f"  • If no real images exist, do NOT mention visuals at all\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"CONTENT IMPROVEMENTS:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"IF CLARITY SCORE < 7:\n"
            f"  • Break up long paragraphs (3-5 sentences max each)\n"
            f"  • Add clear section headers with descriptive titles\n"
            f"  • Use bullet points or numbered lists for multiple items\n"
            f"  • Simplify complex sentences - aim for 15-25 words per sentence\n"
            f"  • Define technical terms on first use\n"
            f"  • Add transitions between sections\n\n"
            f"IF LOGIC SCORE < 7:\n"
            f"  • Find and fix ALL contradictions (cite specific locations)\n"
            f"  • Remove ALL hallucinated content (fake data, fake images, fake sources)\n"
            f"  • Verify dates are accurate and consistent\n"
            f"  • Ensure cause-effect relationships are valid\n"
            f"  • Cross-check all internal references\n"
            f"  • Align claims with provided evidence\n\n"
            f"IF ACTIONABILITY SCORE < 7:\n"
            f"  • Add SPECIFIC examples (not 'for example, things like...')\n"
            f"  • Convert principles to concrete steps\n"
            f"  • Include actual numbers, names, timeframes\n"
            f"  • Replace vague language: 'several' → '3-5', 'often' → '60% of the time'\n"
            f"  • Provide clear next actions or deliverables\n"
            f"  • Make it immediately usable without further research\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"FORMAT & STYLE REQUIREMENTS:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"OUTPUT FORMAT:\n"
            f"  • Return in MARKDOWN format (NOT JSON)\n"
            f"  • Use the Output Schema from the plan as a structural GUIDE\n"
            f"  • Write in clear, natural paragraphs with proper headers\n\n"
            f"MATHEMATICAL NOTATION:\n"
            f"  • Use HTML tags for superscripts: x<sup>2</sup>, 10<sup>6</sup>\n"
            f"  • Use HTML tags for subscripts: H<sub>2</sub>O, CO<sub>2</sub>\n"
            f"  • These render properly in markdown viewers\n\n"
            f"DATE HANDLING:\n"
            f"  • Include dates ONLY when contextually essential\n"
            f"  • Avoid adding dates as mere timestamps\n"
            f"  • Verify accuracy against today's date: {current_date}\n\n"
            f"REFERENCES:\n"
            f"  • Place web links in a 'References' section at the end\n"
            f"  • Format as numbered list with descriptive text\n\n"
            f"META-COMMENTARY:\n"
            f"  ✗ REMOVE: 'This response aims to...', 'I've structured this...', 'The following will...'\n"
            f"  ✗ REMOVE: 'This expanded response...', 'I've added...', 'As requested...'\n"
            f"  ✗ REMOVE: References to the scoring/improvement process\n"
            f"  ✓ KEEP: Just the actual content the user wants to read\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"FINAL CHECKS BEFORE SUBMISSION:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"[ ] Every feedback point has been addressed\n"
            f"[ ] All fake/placeholder images have been removed\n"
            f"[ ] No contradictions remain in the text\n"
            f"[ ] Response directly answers the user's question\n"
            f"[ ] No meta-commentary about the response itself\n"
            f"[ ] Markdown formatting is clean and readable\n"
            f"[ ] Specific examples and concrete details are included\n\n"
            f"Return ONLY the improved final answer in MARKDOWN. No explanations of changes. No JSON wrapper."
        )
        return generate_text(prompt, temperature=0.6)

    def _collate_collected_info(self, collected_info: dict):
        import os
        assets: list[str] = []
        block_payloads = {}

        for key, value in collected_info.items():
            if isinstance(value, dict):
                sanitized = {}
                for inner_key, inner_value in value.items():
                    if inner_key in {"files", "plots"} and isinstance(inner_value, list):
                        # CRITICAL: Only include files that actually exist on disk
                        for item in inner_value:
                            item_path = str(item)
                            # Include asset paths even if files are not present on disk. The UI
                            # will verify existence when attempting to render. Keep a warning
                            # so we can trace missing assets during debugging, but still report
                            # the paths to callers/tests.
                            if os.path.isfile(item_path):
                                assets.append(item_path)
                            else:
                                logger.warning(f"Asset path reported but file not found: {item_path}")
                                assets.append(item_path)
                    sanitized[inner_key] = self._coerce_serializable(inner_value)
                block_payloads[key] = sanitized
            else:
                block_payloads[key] = self._coerce_serializable(value)

        return {"blocks": block_payloads}, assets

    @staticmethod
    def _coerce_serializable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [SynthesizeFinalAnswerBlock._coerce_serializable(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): SynthesizeFinalAnswerBlock._coerce_serializable(v)
                for k, v in value.items()
            }
        return repr(value)


class DataAnalysisBlock(PipelineBlock):
    description = "Analyze structured data (tables, CSV, JSON). Extracts patterns, statistics, correlations, and insights."
    
    def __call__(self, text, data_description) -> dict:
        """
        Analyze structured data to extract insights.
        
        Args:
            text: The analysis request
            data_description: Description of the data or the data itself
            
        Returns:
            dict with analysis results including statistics, patterns, and visualizations
        """
        # Generate analysis plan
        analysis_plan = self.plan_analysis(text, data_description)
        
        # Execute analysis via code
        code_prompt = (
            f"Generate Python code to analyze this data:\n\n"
            f"REQUEST: {text}\n\n"
            f"DATA: {data_description}\n\n"
            f"ANALYSIS PLAN: {analysis_plan}\n\n"
            f"The code must:\n"
            f"1. Parse/load the data (if it's CSV/JSON text, parse it)\n"
            f"2. Calculate statistics (mean, median, std dev, correlations, etc.)\n"
            f"3. Identify patterns or outliers\n"
            f"4. Output results as JSON with 'summary', 'statistics', 'insights' keys\n"
            f"5. Optionally create visualizations with matplotlib if helpful\n\n"
            f"Use only standard library + matplotlib. Return ONLY Python code."
        )
        
        code_block = UseCodeToolBlock(max_attempts=5)
        result = code_block(code_prompt, extract_info=analysis_plan)
        
        return {
            "analysis_plan": analysis_plan,
            "code_result": result,
            "structured_output": result.get("structured_output"),
        }
    
    def plan_analysis(self, text, data_description):
        """Plan what analysis to perform."""
        prompt = (
            f"ROLE: You are a data analyst planning an analysis strategy.\n\n"
            f"USER REQUEST: {text}\n\n"
            f"DATA AVAILABLE: {data_description}\n\n"
            f"Create a concise analysis plan (3-5 bullet points) specifying:\n"
            f"• What statistics to calculate\n"
            f"• What patterns to look for\n"
            f"• What visualizations would be helpful\n"
            f"• What insights to extract\n\n"
            f"Be specific and actionable."
        )
        return generate_text(prompt, temperature=0.5, max_tokens=400)


class ComparisonBlock(PipelineBlock):
    description = "Compare multiple items (products, ideas, options). Provides structured pros/cons and recommendations."
    
    def __call__(self, text, items_to_compare) -> dict:
        """
        Compare multiple items across relevant dimensions.
        
        Args:
            text: The comparison request
            items_to_compare: List or description of items to compare
            
        Returns:
            dict with comparison matrix, pros/cons, and recommendation
        """
        comparison = self.compare_items(text, items_to_compare)
        
        return {
            "comparison_result": comparison,
        }
    
    def compare_items(self, text, items_to_compare):
        """Generate structured comparison."""
        prompt = (
            f"ROLE: You are an expert analyst creating objective comparisons.\n\n"
            f"REQUEST: {text}\n\n"
            f"ITEMS TO COMPARE: {items_to_compare}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"TASK: Create a comprehensive comparison in JSON format.\n\n"
            f"OUTPUT STRUCTURE:\n"
            f"{{\n"
            f"  \"criteria\": [\"criterion1\", \"criterion2\", ...],\n"
            f"  \"items\": [\n"
            f"    {{\n"
            f"      \"name\": \"Item 1\",\n"
            f"      \"scores\": {{\"criterion1\": 8, \"criterion2\": 6, ...}},\n"
            f"      \"pros\": [\"advantage 1\", \"advantage 2\"],\n"
            f"      \"cons\": [\"disadvantage 1\", \"disadvantage 2\"],\n"
            f"      \"best_for\": \"Use case where this excels\"\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"recommendation\": {{\n"
            f"    \"winner\": \"Item name\",\n"
            f"    \"reasoning\": \"Why this is the best choice\",\n"
            f"    \"alternatives\": \"When to consider other options\"\n"
            f"  }}\n"
            f"}}\n\n"
            f"REQUIREMENTS:\n"
            f"  • Use 0-10 scoring for each criterion\n"
            f"  • Be objective - support claims with reasoning\n"
            f"  • Identify clear trade-offs\n"
            f"  • Make a definitive recommendation unless truly tied\n\n"
            f"Return ONLY the JSON object."
        )
        
        raw_response = generate_text(prompt, temperature=0.4, max_tokens=2000)
        
        # Parse JSON response
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines).strip()
            
            return json.loads(cleaned)
        except:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"error": "Could not parse comparison response", "raw": raw_response[:500]}

class SummarizeBlock(PipelineBlock):
    description = "Summarize long documents, articles, or text. Supports different summary styles (bullet points, abstract, ELI5)."
    
    def __call__(self, text, content_to_summarize, summary_style="comprehensive") -> dict:
        """
        Summarize content in the requested style.
        
        Args:
            text: Instructions for summarization
            content_to_summarize: The content to summarize
            summary_style: "comprehensive", "bullets", "abstract", "eli5", or "executive"
            
        Returns:
            dict with summary in requested format
        """
        summary = self.create_summary(text, content_to_summarize, summary_style)
        
        return {
            "original_length": len(content_to_summarize),
            "summary": summary,
            "style": summary_style,
        }
    
    def create_summary(self, instructions, content, style):
        """Generate summary in specified style."""
        style_guides = {
            "comprehensive": "Detailed summary covering all main points, key arguments, and supporting evidence. 3-5 paragraphs.",
            "bullets": "Concise bullet-point list of key takeaways (5-10 bullets). Each bullet is one sentence.",
            "abstract": "Academic-style abstract (150-250 words): background, methods, findings, conclusions.",
            "eli5": "Explain Like I'm 5: Simple language, analogies, no jargon. Make it accessible to a child.",
            "executive": "Executive summary for busy leaders: Bottom line first, key metrics, actionable insights. 2-3 paragraphs max.",
        }
        
        style_guide = style_guides.get(style, style_guides["comprehensive"])
        
        prompt = (
            f"ROLE: You are an expert at distilling complex content into clear summaries.\n\n"
            f"INSTRUCTIONS: {instructions}\n\n"
            f"CONTENT TO SUMMARIZE:\n{content[:15000]}\n\n"  # Limit to prevent token overflow
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"SUMMARY STYLE: {style}\n"
            f"REQUIREMENTS: {style_guide}\n\n"
            f"CRITICAL RULES:\n"
            f"  • Capture the MAIN ideas accurately - no hallucinations\n"
            f"  • Use the author's terminology for key concepts\n"
            f"  • Preserve important numbers, dates, and names\n"
            f"  • Maintain objectivity - don't add your opinions\n"
            f"  • If content is too short to summarize, say so\n\n"
            f"Provide the summary now (do NOT wrap in JSON):"
        )
        
        return generate_text(prompt, temperature=0.4, max_tokens=1000)


class TranslateBlock(PipelineBlock):
    description = "Translate text between languages while preserving tone, style, and cultural context."
    
    def __call__(self, text, content_to_translate, target_language, source_language="auto") -> dict:
        """
        Translate content to target language.
        
        Args:
            text: Any additional context or instructions
            content_to_translate: The text to translate
            target_language: Target language (e.g., "Spanish", "French", "Japanese")
            source_language: Source language or "auto" for auto-detection
            
        Returns:
            dict with translated text and metadata
        """
        translation = self.translate_content(text, content_to_translate, target_language, source_language)
        
        return {
            "original": content_to_translate[:500],  # Preview
            "translation": translation,
            "target_language": target_language,
        }
    
    def translate_content(self, context, content, target_lang, source_lang):
        """Perform translation with cultural awareness."""
        prompt = (
            f"ROLE: You are an expert translator fluent in {target_lang}.\n\n"
            f"CONTEXT: {context}\n\n"
            f"SOURCE LANGUAGE: {source_lang}\n"
            f"TARGET LANGUAGE: {target_lang}\n\n"
            f"CONTENT TO TRANSLATE:\n{content}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"TRANSLATION REQUIREMENTS:\n"
            f"  • Preserve the original meaning and tone precisely\n"
            f"  • Adapt idioms and cultural references appropriately\n"
            f"  • Maintain formality level (casual vs formal)\n"
            f"  • Keep technical terms accurate\n"
            f"  • Use natural phrasing in target language (not word-for-word)\n"
            f"  • Preserve formatting (line breaks, emphasis, etc.)\n\n"
            f"Provide ONLY the translated text (no explanations or meta-commentary):"
        )
        
        return generate_text(prompt, temperature=0.3, max_tokens=2000)


class FactCheckBlock(PipelineBlock):
    description = "Verify factual claims using web search and logical analysis. Returns verification status for each claim."
    
    def __call__(self, text, claims_to_verify) -> dict:
        """
        Fact-check claims using web search and reasoning.
        
        Args:
            text: Context for the fact-check
            claims_to_verify: String or list of claims to verify
            
        Returns:
            dict with verification results for each claim
        """
        # Parse claims into list
        if isinstance(claims_to_verify, str):
            claims_list = [c.strip() for c in claims_to_verify.split('\n') if c.strip()]
        else:
            claims_list = claims_to_verify
        
        # Verify each claim
        results = []
        for i, claim in enumerate(claims_list[:5]):  # Limit to 5 claims to avoid quota issues
            logger.info(f"Fact-checking claim {i+1}: {claim[:100]}...")
            verification = self.verify_claim(claim, text)
            results.append({
                "claim": claim,
                "verification": verification,
            })
        
        return {
            "claims_checked": len(results),
            "results": results,
        }
    
    def verify_claim(self, claim, context):
        """Verify a single claim."""
        # First, try web search for factual verification
        search_query = f"verify fact: {claim}"
        web_result = online_query_tool(search_query, num_links=3)
        
        # Then use LLM to analyze the evidence
        prompt = (
            f"ROLE: You are a fact-checker analyzing evidence for a claim.\n\n"
            f"CLAIM TO VERIFY: {claim}\n\n"
            f"CONTEXT: {context}\n\n"
            f"WEB SEARCH RESULTS:\n{web_result.get('summary', 'No results')}\n\n"
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"TASK: Determine if the claim is TRUE, FALSE, PARTIALLY_TRUE, or UNVERIFIABLE.\n\n"
            f"OUTPUT FORMAT (JSON only):\n"
            f"{{\n"
            f"  \"verdict\": \"TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIABLE\",\n"
            f"  \"confidence\": \"high/medium/low\",\n"
            f"  \"evidence\": \"Supporting or contradicting evidence found\",\n"
            f"  \"reasoning\": \"Why you reached this verdict\",\n"
            f"  \"sources\": [\"source1\", \"source2\"]\n"
            f"}}\n\n"
            f"CRITICAL RULES:\n"
            f"  • If web search failed, mark as UNVERIFIABLE with low confidence\n"
            f"  • Do NOT guess or use general knowledge for specific factual claims\n"
            f"  • Be conservative - if uncertain, say UNVERIFIABLE\n"
            f"  • Show your reasoning clearly\n\n"
            f"Return ONLY the JSON object."
        )
        
        raw_response = generate_text(prompt, temperature=0.2, max_tokens=600)
        
        # Parse response
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines).strip()
            
            return json.loads(cleaned)
        except:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": "low",
                "evidence": "Could not parse verification response",
                "reasoning": "Technical error in fact-checking",
                "sources": []
            }


class DebugBlock(PipelineBlock):
    description = "Debug code, identify issues, and suggest fixes. Supports Python, JavaScript, and other languages."
    
    def __call__(self, text, code_to_debug, error_message=None) -> dict:
        """
        Debug code and suggest fixes.
        
        Args:
            text: Description of the problem
            code_to_debug: The code with issues
            error_message: Optional error message or stack trace
            
        Returns:
            dict with diagnosis, fixes, and corrected code
        """
        diagnosis = self.diagnose_code(text, code_to_debug, error_message)
        
        return {
            "diagnosis": diagnosis,
        }
    
    def diagnose_code(self, problem_description, code, error_message):
        """Diagnose code issues and provide fixes."""
        prompt = (
            f"ROLE: You are an expert debugger analyzing problematic code.\n\n"
            f"PROBLEM DESCRIPTION: {problem_description}\n\n"
            f"CODE WITH ISSUES:\n```\n{code}\n```\n\n"
        )
        
        if error_message:
            prompt += f"ERROR MESSAGE:\n{error_message}\n\n"
        
        prompt += (
            f"═══════════════════════════════════════════════════════════════════════════════════\n"
            f"TASK: Identify all issues and provide fixes.\n\n"
            f"OUTPUT FORMAT (JSON only):\n"
            f"{{\n"
            f"  \"issues_found\": [\n"
            f"    {{\n"
            f"      \"type\": \"syntax/logic/performance/security\",\n"
            f"      \"severity\": \"critical/major/minor\",\n"
            f"      \"location\": \"Line 5 or function name\",\n"
            f"      \"description\": \"What's wrong\",\n"
            f"      \"fix\": \"How to fix it\"\n"
            f"    }}\n"
            f"  ],\n"
            f"  \"corrected_code\": \"Full corrected version of the code\",\n"
            f"  \"explanation\": \"Summary of changes made\",\n"
            f"  \"testing_suggestions\": [\"How to test the fix\"]\n"
            f"}}\n\n"
            f"REQUIREMENTS:\n"
            f"  • Identify ALL issues, not just the first one\n"
            f"  • Provide working, tested fixes\n"
            f"  • Explain WHY each change is necessary\n"
            f"  • Consider edge cases and best practices\n\n"
            f"Return ONLY the JSON object."
        )
        
        raw_response = generate_text(prompt, temperature=0.3, max_tokens=2000)
        
        # Parse response
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = '\n'.join(lines).strip()
            
            return json.loads(cleaned)
        except:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {"error": "Could not parse debug response", "raw": raw_response[:500]}


class ReadPDFBlock(PipelineBlock):
    description = "Extract text content from uploaded PDF files for analysis."
    
    def __call__(self, text, pdf_path) -> dict:
        """
        Extract text from a PDF file.
        
        Args:
            text: Context or instructions
            pdf_path: Path to the PDF file
            
        Returns:
            dict with extracted text and metadata
        """
        extracted_text = extract_text_from_pdf(pdf_path)
        
        return {
            "pdf_path": pdf_path,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "success": not extracted_text.startswith("ERROR:")
        }


class CreatePDFBlock(PipelineBlock):
    description = "Create PDF documents from text or markdown content. Useful for generating reports, summaries, or formatted documents."
    
    def __call__(self, text, content_to_convert, output_filename="output.pdf", title=None, use_markdown=True) -> dict:
        """
        Create a PDF from text content.
        
        Args:
            text: Context or instructions
            content_to_convert: The text/markdown content to convert to PDF
            output_filename: Name for the output PDF file
            title: Optional title for the PDF document
            use_markdown: Whether to treat content as markdown (True) or plain text (False)
            
        Returns:
            dict with PDF creation results and file path
        """
        import tempfile
        import os
        
        # Create output path in temp directory
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, output_filename)
        
        # Create PDF based on format
        if use_markdown:
            success = create_pdf_from_markdown(content_to_convert, output_path, title)
        else:
            success = create_pdf_from_text(content_to_convert, output_path, title)
        
        result = {
            "success": success,
            "output_path": output_path if success else None,
            "filename": output_filename,
            "title": title,
            "format": "markdown" if use_markdown else "plain_text",
            "content_length": len(content_to_convert)
        }
        
        if success:
            logger.info(f"PDF created successfully: {output_path}")
        else:
            logger.error(f"Failed to create PDF: {output_filename}")
        
        return result
