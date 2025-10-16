import json
import logging
import re
from typing import Optional

from .utility import generate_text, python_exec_tool, online_query_tool


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
    description = "Search the web. Requires data.search_query and optional data.link_num (1-10)."
    
    def __call__(self, query, num_links):
        output = online_query_tool(query, num_links)
        summary = output.get("summary", "")
        articles = output.get("articles", [])
        text = f"""The output to the requested information {query} from the internet search is:

{summary}

Links used:

{articles}

        """
        return text


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
        initial_response = self.create_synthesis_prompt(prompt, plan, collected_info)
        scorer = self.scorer_prompt(prompt, initial_response)
        improved = self.improver_prompt(prompt, initial_response, plan, scorer)
        return {
            "initial_response": initial_response,
            "score": scorer,
            "final_response": improved,
        }

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
            f"9. Be factual and precise - do not hallucinate data, sources, or images that don't exist\n"
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
