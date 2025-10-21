"""Thinking pipeline implementation.

This module provides the high-level orchestration logic that wires
pipeline blocks together to answer user queries. The `ThinkingPipeline`
class is responsible for

1. Planning: determining which blocks should run for a prompt.
2. Execution: running blocks sequentially while sharing context.
3. State management: storing intermediate results and logs.
4. Final synthesis: producing the final response for the user.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .pipeline_blocks import (
    PlanCreationBlock,
    UseCodeToolBlock,
    UseInternetToolBlock,
    MathImprovementBlock,
    CreativeIdeaGeneratorBlockTool,
    SynthesizeFinalAnswerBlock,
)
from .utility import generate_text


logger = logging.getLogger(__name__)


class ThinkingPipeline:
    """High-level orchestrator for the reasoning pipeline."""

    ROUTER_SYSTEM_PROMPT = (
        "You are a pipeline architect selecting optimal processing blocks. "
        "Design efficient sequences that balance thoroughness with resource constraints. "
        "Always end with 'synthesize_final_answer'."
    )

    def __init__(self, verbose: bool = False, progress_callback: Optional[Callable] = None) -> None:
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.plan_block = PlanCreationBlock()
        self.block_registry: Dict[str, Any] = {
            "plan_creation": self.plan_block,
            "use_code_tool": UseCodeToolBlock(),
            "use_internet_tool": UseInternetToolBlock(),
            "math_improvement": MathImprovementBlock(),
            "creative_idea_generator": CreativeIdeaGeneratorBlockTool(),
            "synthesize_final_answer": SynthesizeFinalAnswerBlock(),
        }
        self.state: Dict[str, Any] = {}
        
        # Build block descriptions dynamically from block classes
        self.ROUTER_BLOCK_DESCRIPTION = self._build_block_descriptions()
    
    def _build_block_descriptions(self) -> str:
        """Build router block descriptions from block class descriptions."""
        descriptions = ["Available blocks:"]
        for key, block in self.block_registry.items():
            desc = getattr(block, 'description', f"{key} block")
            descriptions.append(f"- {key}: {desc}")
        return "\n".join(descriptions) + "\n"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def reset_state(self, prompt: str) -> None:
        self.state = {
            "prompt": prompt,
            "plan": None,
            "pipeline": [],
            "context": {},
            "logs": [],
            "final_answer": None,
            "assets": [],
        }

    def log(self, stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        log_entry = {"stage": stage, "message": message, "data": data or {}}
        self.state.setdefault("logs", []).append(log_entry)
        if self.verbose:
            logger.info("[%s] %s -- %s", stage, message, data)
    
    def _build_context_summary(self, context: Dict[str, Any]) -> str:
        """Build a concise summary of previous block outputs for context sharing."""
        if not context:
            return ""
        
        summary_parts = []
        for key, value in context.items():
            # Extract just the block name (remove _index suffix)
            block_name = key.rsplit('_', 1)[0]
            
            # Summarize the output based on type
            if isinstance(value, dict):
                # For dicts, show key fields
                if "answer" in value:
                    summary_parts.append(f"- {block_name}: {value['answer'][:200]}")
                elif "summary" in value:
                    summary_parts.append(f"- {block_name}: {value['summary'][:200]}")
                elif "ideas" in value:
                    summary_parts.append(f"- {block_name}: Generated ideas")
                else:
                    summary_parts.append(f"- {block_name}: {str(value)[:200]}")
            elif isinstance(value, str):
                summary_parts.append(f"- {block_name}: {value[:200]}")
            else:
                summary_parts.append(f"- {block_name}: {str(value)[:200]}")
        
        return "\n".join(summary_parts) if summary_parts else ""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def determine_pipeline(self, prompt: str) -> List[Dict[str, Any]]:
        """Create an internal plan and determine block sequence."""

        if self.progress_callback:
            self.progress_callback(0, 3, "Creating execution plan")

        self.log("planning", "Generating execution plan")
        plan_text = self.plan_block(prompt)
        self.state["plan"] = plan_text

        if self.progress_callback:
            self.progress_callback(1, 3, "Routing to blocks")

        router_prompt = self._build_router_prompt(prompt, plan_text)
        # Increased max_tokens from 600 to 2048 to prevent JSON truncation with multiple blocks
        raw_router_response = generate_text(router_prompt, temperature=0.2, max_tokens=2048)
        router_payload = self._parse_router_response(raw_router_response)

        pipeline_spec = router_payload.get("pipeline_blocks", [])
        pipeline_spec = self._validate_pipeline_spec(pipeline_spec)
        self.state["pipeline"] = pipeline_spec
        self.log("planning", "Pipeline determined", {"pipeline": pipeline_spec})
        return pipeline_spec

    def _build_router_prompt(self, prompt: str, plan: str) -> str:
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        return (
            f"Today's date: {current_date}. Use this for time-sensitive routing decisions.\n\n"
            f"ROLE: You are an expert pipeline architect designing optimal execution sequences.\n\n"
            f"OBJECTIVE: Select the minimal, most effective sequence of processing blocks to answer the user's request.\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"USER REQUEST:\n{prompt}\n\n"
            f"EXECUTION PLAN (generated earlier):\n{plan}\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"AVAILABLE BLOCKS:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"{self.ROUTER_BLOCK_DESCRIPTION}\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"PIPELINE DESIGN PRINCIPLES:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"EFFICIENCY (CRITICAL):\n"
            f"  • TARGET: 3-5 blocks total (including synthesis)\n"
            f"  • MAXIMUM: 7 blocks (hard API quota limit)\n"
            f"  • Each block = 1-3 API calls = significant token cost\n"
            f"  • Quality > Quantity - fewer well-chosen blocks beat many weak ones\n"
            f"  • Ask: 'Is this block truly necessary or just nice-to-have?'\n\n"
            f"BLOCK SELECTION STRATEGY:\n"
            f"  • Only include blocks that DIRECTLY contribute to answering the request\n"
            f"  • Avoid redundancy - one internet search is usually enough\n"
            f"  • Skip blocks that duplicate information already available\n"
            f"  • For simple queries, use just synthesis (no intermediate blocks)\n"
            f"  • For complex queries, chain blocks that build on each other\n\n"
            f"COMMON PATTERNS:\n"
            f"  • Data analysis: code_tool → synthesis (2 blocks)\n"
            f"  • Research: internet_tool → synthesis (2 blocks)\n"
            f"  • Creative work: creative_ideas → synthesis (2 blocks)\n"
            f"  • Math validation: math_improvement → synthesis (2 blocks)\n"
            f"  • Math with computation: code_tool → math_improvement → synthesis (3 blocks)\n"
            f"  • Complex math: internet_tool → code_tool → math_improvement → synthesis (4 blocks)\n"
            f"  • Complex: internet_tool → code_tool → synthesis (3 blocks)\n"
            f"  • Brainstorming: creative_ideas (choose_best=true) → synthesis (2 blocks)\n"
            f"  • Research + Math: internet_tool → math_improvement → synthesis (3 blocks)\n\n"
            f"MANDATORY RULES:\n"
            f"  ✓ ALWAYS end with 'synthesize_final_answer' (exactly once, last position)\n"
            f"  ✓ If using creative_idea_generator for brainstorming: set choose_best=true\n"
            f"  ✓ If user wants visualizations: include use_code_tool with plot instructions\n"
            f"  ✗ NEVER include plan_creation (already executed)\n"
            f"  ✗ NEVER repeat the same block type twice (except in rare cases)\n"
            f"  ✗ NEVER exceed 7 blocks total\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"BLOCK DATA SPECIFICATIONS:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"use_code_tool:\n"
            f"  Required: {{\"extract_info\": \"<specific instructions for what to compute/analyze/plot>\"}}\n"
            f"  Example: {{\"extract_info\": \"Calculate mean and median, generate histogram plot\"}}\n"
            f"  Use when: Need computations, data analysis, or visualizations\n\n"
            f"use_internet_tool:\n"
            f"  Required: {{\"search_query\": \"<search terms>\", \"link_num\": <1-10>}}\n"
            f"  Example: {{\"search_query\": \"latest climate data 2025\", \"link_num\": 5}}\n"
            f"  Use when: Need current information, research, or fact-checking\n"
            f"  Note: link_num defaults to 3 if omitted\n"
            f"  Critical: If fetch fails, returns explicit error - do NOT hallucinate web content\n"
            f"  QUERY IMPROVEMENT:\n"
            f"    • If user says 'what is the news today': transform to 'news today', 'breaking news', 'current events'\n"
            f"    • If user says 'news <date>': use exact date in query (e.g., 'news October 20 2025')\n"
            f"    • If query is too vague: add specificity (e.g., 'weather' → 'current weather today')\n"
            f"    • For time-sensitive queries: always include the date context\n"
            f"    • Use single keywords when possible: 'latest AI announcements' not 'what are the announcements'\n\n"
            f"math_improvement:\n"
            f"  Required: {{\"math_content\": \"<mathematical reasoning/calculation to validate>\"}}\n"
            f"  Example: {{\"math_content\": \"Calculate integral of 2x^2 + 3x from 0 to 5\"}}\n"
            f"  Use when: Need to verify math, check for algebraic/calculus errors, validate logic\n"
            f"  Returns: Validation report with error detection and corrections if needed\n"
            f"  Critical: Only flags errors that can be mathematically verified - never hallucinate\n"
            f"  IMPORTANT FOR COMPLEX MATH: If calculation requires numerical precision or involves:\n"
            f"    • Multi-step computations that need verification\n"
            f"    • Numerical integration, differentiation, or solving equations\n"
            f"    • Data processing or statistical analysis\n"
            f"    • Matrix operations, linear algebra, or complex formulas\n"
            f"    → Route to use_code_tool FIRST for computation, THEN math_improvement for validation\n"
            f"    → Pattern: code_tool (compute) → math_improvement (validate) → synthesis\n"
            f"    → This ensures accurate numerical results that are then verified for correctness\n\n"
            f"creative_idea_generator:\n"
            f"  Required: {{\"criteria\": \"<what makes an idea good for this request>\"}}\n"
            f"  Optional: {{\"choose_best\": true/false}}\n"
            f"  Example: {{\"criteria\": \"Original, feasible, emotionally resonant\", \"choose_best\": true}}\n"
            f"  Use when: Brainstorming, creative work, generating options\n"
            f"  IMPORTANT: Set choose_best=true to auto-select and expand the best idea\n\n"
            f"synthesize_final_answer:\n"
            f"  Data: {{}}\n"
            f"  Use: ALWAYS include as the final block\n"
            f"  Purpose: Combines all previous outputs into a polished final answer\n\n"
            f"═════════════════════════════════════════════════════════════════\n"
            f"OUTPUT FORMAT - EXACT JSON REQUIRED:\n"
            f"═════════════════════════════════════════════════════════════════\n\n"
            f"Return a JSON object with two keys:\n\n"
            f"{{\n"
            f"  \"justification\": \"<1-2 sentences explaining your block choices and sequence>\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"<block_name>\", \"data\": {{...}}}},\n"
            f"    ...\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"EXAMPLE 1 (Simple data analysis):\n"
            f"{{\n"
            f"  \"justification\": \"Code tool generates the requested statistics and plot, synthesis delivers results.\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"use_code_tool\", \"data\": {{\"extract_info\": \"Calculate summary stats, create bar chart\"}}}},\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"EXAMPLE 2 (Creative brainstorming with selection):\n"
            f"{{\n"
            f"  \"justification\": \"Creative block generates poem ideas and selects best, synthesis polishes.\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"creative_idea_generator\", \"data\": {{\"criteria\": \"Vivid imagery, emotional depth\", \"choose_best\": true}}}},\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"EXAMPLE 3 (Research + analysis):\n"
            f"{{\n"
            f"  \"justification\": \"Internet tool gathers data, code tool analyzes trends, synthesis presents findings.\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"use_internet_tool\", \"data\": {{\"search_query\": \"renewable energy 2025\", \"link_num\": 5}}}},\n"
            f"    {{\"key\": \"use_code_tool\", \"data\": {{\"extract_info\": \"Extract key statistics, create trend visualization\"}}}},\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"EXAMPLE 4 (Math validation and correction):\n"
            f"{{\n"
            f"  \"justification\": \"Math improvement block validates the mathematical reasoning, synthesis explains results.\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"math_improvement\", \"data\": {{\"math_content\": \"Solve: 2x^2 + 5x - 3 = 0 using quadratic formula\"}}}},\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"EXAMPLE 5 (Complex math with numerical computation and validation):\n"
            f"{{\n"
            f"  \"justification\": \"Code tool performs precise numerical calculations, math improvement validates methodology and results, synthesis presents findings.\",\n"
            f"  \"pipeline_blocks\": [\n"
            f"    {{\"key\": \"use_code_tool\", \"data\": {{\"extract_info\": \"Compute definite integral of f(x) = 2x^2 + 3x from 0 to 5 using numerical methods\"}}}},\n"
            f"    {{\"key\": \"math_improvement\", \"data\": {{\"math_content\": \"Validate the integral calculation and check if the numerical method was applied correctly\"}}}},\n"
            f"    {{\"key\": \"synthesize_final_answer\", \"data\": {{}}}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"Remember: Keep it short, smart, and purposeful. Every block must earn its place in the pipeline."
        )

    def _parse_router_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON payload from the router model response."""

        self.log("planning", "Parsing router response", {"raw": response})
        cleaned = response.strip()
        
        # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            lines = cleaned.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]  # Remove first line
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            cleaned = '\n'.join(lines).strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            self.log("planning", "Router response not valid JSON; attempting recovery", None)

        extracted = self._extract_json_text(cleaned)
        for candidate in self._generate_json_candidates(extracted):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    coerced = ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    continue
                else:
                    if isinstance(coerced, dict):
                        return coerced

        self.log("planning", "Router response parsing failed", {"raw": response})
        raise ValueError("Unable to parse router response as JSON")

    @staticmethod
    def _extract_json_text(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Router response did not contain JSON")
        return text[start : end + 1]

    @staticmethod
    def _generate_json_candidates(text: str) -> List[str]:
        candidates = [text]
        candidates.append(text.replace("'", '"'))
        candidates.append(text.replace("True", "true").replace("False", "false"))
        candidates.append(
            text.replace("True", "true").replace("False", "false").replace("'", '"')
        )
        # Defensive: candidates may contain None; ensure strings before strip()
        return [str(c).strip() for c in candidates if c]

    def _validate_pipeline_spec(self, spec: Any) -> List[Dict[str, Any]]:
        if not isinstance(spec, list):
            raise ValueError("pipeline_blocks must be a list")
        
        # CRITICAL: Limit pipeline length to prevent quota exhaustion
        # Each block typically makes 1-3 API calls. With 15K tokens/min quota:
        # - Average call: ~2-3K tokens
        # - Max safe calls per minute: ~5-6 calls
        # - ABSOLUTE MAX: 7 blocks (not a target, but safety limit)
        # - IDEAL: 3-5 blocks for most tasks
        MAX_PIPELINE_BLOCKS = 7
        if len(spec) > MAX_PIPELINE_BLOCKS:
            logger.warning(
                f"Pipeline has {len(spec)} blocks, exceeding MAXIMUM SAFE LIMIT of {MAX_PIPELINE_BLOCKS}. "
                f"Truncating to prevent quota exhaustion. "
                f"NOTE: Shorter pipelines (3-5 blocks) are generally better and more efficient."
            )
            # Keep only the first blocks and ensure synthesis is last
            spec = spec[:MAX_PIPELINE_BLOCKS]
            # Ensure synthesize_final_answer is at the end
            if spec[-1].get("key") != "synthesize_final_answer":
                # Remove any synthesis blocks from middle
                spec = [b for b in spec if b.get("key") != "synthesize_final_answer"]
                # Add synthesis at the end
                spec = spec[:MAX_PIPELINE_BLOCKS-1] + [{"key": "synthesize_final_answer", "data": {}}]

        validated: List[Dict[str, Any]] = []
        seen_final = False

        for idx, entry in enumerate(spec):
            if not isinstance(entry, dict) or "key" not in entry:
                raise ValueError(f"Invalid pipeline entry at index {idx}: {entry}")

            key = entry["key"]
            if key not in self.block_registry:
                raise ValueError(f"Unknown pipeline block key: {key}")

            data = entry.get("data", {})
            if not isinstance(data, dict):
                raise ValueError(f"data for block {key} must be a dict")

            if key == "synthesize_final_answer":
                seen_final = True
            elif key == "plan_creation" and idx != 0:
                self.log(
                    "validation",
                    "plan_creation block is not first; moving to front",
                )

            validated.append({"key": key, "data": data})

        if not validated or validated[-1]["key"] != "synthesize_final_answer":
            raise ValueError("Pipeline must end with synthesize_final_answer")

        if not seen_final:
            raise ValueError("Pipeline missing synthesize_final_answer block")

        return validated

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_pipeline(self, pipeline_spec: List[Dict[str, Any]]) -> None:
        context: Dict[str, Any] = {}
        total_blocks = len(pipeline_spec)

        for idx, block_info in enumerate(pipeline_spec):
            key = block_info["key"]
            block = self.block_registry[key]
            data = block_info.get("data", {})

            # Report progress if callback provided
            if self.progress_callback:
                self.progress_callback(idx, total_blocks, key)

            self.log("execution", f"Running block '{key}'", {"data": data, "index": idx})

            # Build context summary from previous blocks for this block to access
            previous_context_summary = self._build_context_summary(context)
            
            if key == "plan_creation":
                result = self.state.get("plan")
            elif key == "use_code_tool":
                src_text = data.get("input_text", self.state["prompt"])
                # Add previous context if available
                if previous_context_summary:
                    src_text = f"{src_text}\n\n[PREVIOUS BLOCK OUTPUTS]:\n{previous_context_summary}"
                result = block(src_text, data.get("extract_info"))
            elif key == "use_internet_tool":
                query = data.get("search_query") or self.state["prompt"]
                link_num = int(data.get("link_num", 3))
                result = block(query, link_num)
            elif key == "math_improvement":
                # Get math content from data
                math_content = data.get("math_content", "")
                
                # Build context text including previous block outputs
                context_text = self.state["prompt"]
                if previous_context_summary:
                    context_text = f"{context_text}\n\n[PREVIOUS BLOCK OUTPUTS]:\n{previous_context_summary}"
                
                result = block(context_text, math_content)
            elif key == "creative_idea_generator":
                criteria = data.get("criteria") or "Generate diverse, high-quality ideas."
                choose_best = bool(data.get("choose_best", False))
                source_text = data.get("input_text", self.state["prompt"])
                # Add previous context if available
                if previous_context_summary:
                    source_text = f"{source_text}\n\n[PREVIOUS BLOCK OUTPUTS]:\n{previous_context_summary}"
                result = block(source_text, criteria, choose_best)
            elif key == "synthesize_final_answer":
                result = block(self.state["prompt"], self.state["plan"], context)
                self.state["final_answer"] = result.get("final_response")
            else:
                result = block(self.state["prompt"])

            context_key = f"{key}_{idx}"
            context[context_key] = result
            self.log("execution", f"Block '{key}' completed", {"result_key": context_key})

            if isinstance(result, dict):
                asset_list = []
                for asset_key in ("files", "plots"):
                    assets = result.get(asset_key)
                    if assets:
                        if isinstance(assets, list):
                            asset_list.extend(str(a) for a in assets)
                        else:
                            asset_list.append(str(assets))
                if asset_list:
                    self.state.setdefault("assets", []).extend(asset_list)

        self.state["context"] = context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_context(self) -> Dict[str, Any]:
        return {
            "prompt": self.state.get("prompt"),
            "plan": self.state.get("plan"),
            "pipeline": self.state.get("pipeline"),
            "context": self.state.get("context"),
            "logs": self.state.get("logs"),
            "final_answer": self.state.get("final_answer"),
            "assets": self.state.get("assets", []),
        }

    def __call__(self, prompt: str) -> Dict[str, Any]:
        self.reset_state(prompt)
        pipeline_spec = self.determine_pipeline(prompt)
        self.execute_pipeline(pipeline_spec)
        final_payload = self.collect_context()

        if final_payload.get("final_answer") is None:
            self.log("execution", "Final answer missing after execution")
        return final_payload


__all__ = ["ThinkingPipeline"]
