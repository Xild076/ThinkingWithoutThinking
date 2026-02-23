from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator, Literal

try:
    from pipeline_blocks import (
        CreativeIdeaGeneratorToolBlock,
        DeductiveReasoningToolBlock,
        ImprovementBlock,
        InitialPlanCreationBlock,
        LargeResponseRouterBlock,
        LongResponseSynthesisBlock,
        PrimaryToolRouterBlock,
        PythonCodeExecutionToolBlock,
        SelfCritiqueBlock,
        SubPlanCreationBlock,
        SubToolRouterBlock,
        SynthesisBlock,
        WebSearchToolBlock,
        WikipediaSearchToolBlock,
        _get_all_tool_classes,
    )
    from utility import logger
except ImportError:  # pragma: no cover - fallback for package import mode
    from src.pipeline_blocks import (
        CreativeIdeaGeneratorToolBlock,
        DeductiveReasoningToolBlock,
        ImprovementBlock,
        InitialPlanCreationBlock,
        LargeResponseRouterBlock,
        LongResponseSynthesisBlock,
        PrimaryToolRouterBlock,
        PythonCodeExecutionToolBlock,
        SelfCritiqueBlock,
        SubPlanCreationBlock,
        SubToolRouterBlock,
        SynthesisBlock,
        WebSearchToolBlock,
        WikipediaSearchToolBlock,
        _get_all_tool_classes,
    )
    from src.utility import logger


ThinkingLevel = Literal["low", "med-synth", "med-plan", "high"]


class Pipeline:
    def __init__(
        self,
        tools: list[Any] | None = None,
        max_context_chars: int = 18000,
        compression_keep_recent: int = 6,
        compression_every: int = 3,
        enable_image_embedding: bool = True,
        allow_visual_outputs: bool = True,
    ):
        self.tools = tools or [
            WebSearchToolBlock(),
            WikipediaSearchToolBlock(),
            PythonCodeExecutionToolBlock(),
            CreativeIdeaGeneratorToolBlock(),
            DeductiveReasoningToolBlock(),
        ]
        self._tool_lookup = {tool.details["id"]: tool for tool in self.tools}
        self.max_context_chars = max_context_chars
        self.compression_keep_recent = compression_keep_recent
        self.compression_every = compression_every
        self.enable_image_embedding = bool(enable_image_embedding)
        self.allow_visual_outputs = bool(allow_visual_outputs)
        self.event_history: list[dict[str, Any]] = []

    def _normalize_thinking_level(self, thinking_level: str) -> ThinkingLevel:
        normalized = (thinking_level or "").strip().lower()
        aliases = {
            "medium": "med-synth",
            "medium-synth": "med-synth",
            "medium_synth": "med-synth",
            "med_synth": "med-synth",
            "medium-plan": "med-plan",
            "medium_plan": "med-plan",
            "med_plan": "med-plan",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"low", "med-synth", "med-plan", "high"}:
            normalized = "med-synth"
        return normalized  # type: ignore[return-value]

    def _available_tools(self) -> list[dict[str, Any]]:
        available = []
        for tool in self.tools:
            available.append(
                {
                    "id": tool.details["id"],
                    "name": tool.details["name"],
                    "description": tool.details["description"],
                    "inputs": [
                        {
                            "name": input_obj.name,
                            "description": input_obj.description,
                            "type": input_obj.type,
                        }
                        for input_obj in tool.details["inputs"]
                    ],
                    "outputs": [
                        {
                            "name": output_obj.name,
                            "description": output_obj.description,
                            "type": output_obj.type,
                        }
                        for output_obj in tool.details["outputs"]
                    ],
                }
            )
        return available

    def _plan_review_enabled(self, thinking_level: ThinkingLevel) -> bool:
        return thinking_level in ("med-plan", "high")

    def _synth_review_enabled(self, thinking_level: ThinkingLevel) -> bool:
        return thinking_level in ("med-synth", "high")

    def _looks_like_reading_level(self, value: str) -> bool:
        cleaned = (value or "").strip().lower()
        if not cleaned:
            return False
        if any(token in cleaned for token in ("grade", "reading", "college", "high school", "middle school")):
            return True
        return bool(re.search(r"\b\d+(st|nd|rd|th)\s+grade\b", cleaned))

    def _normalize_audience_fields(self, plan_output: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan_output, dict):
            return {
                "assumed_audience": "General audience",
                "assumed_audience_knowledge_level": "Mixed",
                "assumed_audience_reading_level": "General",
            }

        audience = str(plan_output.get("assumed_audience", "")).strip()
        knowledge = str(plan_output.get("assumed_audience_knowledge_level", "")).strip()
        reading = str(plan_output.get("assumed_audience_reading_level", "")).strip()

        if self._looks_like_reading_level(audience) and not reading:
            reading = audience
            audience = "General audience"

        if not audience:
            audience = "General audience"
        if not knowledge:
            knowledge = "Mixed"
        if not reading:
            reading = "General"

        plan_output["assumed_audience"] = audience
        plan_output["assumed_audience_knowledge_level"] = knowledge
        plan_output["assumed_audience_reading_level"] = reading
        return plan_output

    def _planning_snapshot_payload(
        self,
        complex_response: bool,
        long_response: bool,
        steps: list[str],
        response_criteria: list[str],
        plan_output: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "complex_response": complex_response,
            "long_response": long_response,
            "steps_count": len(steps),
            "response_criteria_count": len(response_criteria),
            "assumed_audience": plan_output.get("assumed_audience", "Unknown"),
            "assumed_audience_knowledge_level": plan_output.get(
                "assumed_audience_knowledge_level", "Unknown"
            ),
            "assumed_audience_reading_level": plan_output.get(
                "assumed_audience_reading_level", "Unknown"
            ),
        }

    def _plan_metadata_payload(
        self,
        plan_text: str,
        steps: list[str],
        response_criteria: list[str],
        plan_output: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "plan": plan_text,
            "steps": steps,
            "response_criteria": response_criteria,
            "assumed_audience": plan_output.get("assumed_audience", "General audience"),
            "assumed_audience_knowledge_level": plan_output.get(
                "assumed_audience_knowledge_level", "Mixed"
            ),
            "assumed_audience_reading_level": plan_output.get(
                "assumed_audience_reading_level", "General"
            ),
        }

    def _ensure_naturalism_criterion(self, response_criteria: list[str]) -> list[str]:
        criteria = [str(item).strip() for item in response_criteria if str(item).strip()]
        if not criteria:
            criteria = []

        lowered = " ".join(criteria).lower()
        natural_tokens = (
            "natural",
            "audience-appropriate",
            "human",
            "conversational",
            "plain language",
            "non-robotic",
            "no meta",
            "jargon",
        )
        if any(token in lowered for token in natural_tokens):
            return criteria

        criteria.append(
            "Final response uses natural, audience-appropriate prose and avoids internal pipeline jargon."
        )
        return criteria

    def _serialize_compact(self, value: Any, max_chars: int = 1200) -> str:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str)
        except Exception:
            text = str(value)
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _build_synthesis_critique_context(
        self,
        prompt: str,
        plan_text: str,
        steps: list[str],
        response_criteria: list[str],
        plan_output: dict[str, Any],
        tool_context: dict[str, Any],
        tool_failure_signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        primary_results = self._extract_results_from_bundle(tool_context.get("primary_execution") or {})
        continuity_results = self._extract_results_from_bundle(
            tool_context.get("primary_continuity_execution") or {}
        )

        subplan_summaries: list[dict[str, Any]] = []
        for subplan in (tool_context.get("subplans") or [])[-3:]:
            if not isinstance(subplan, dict):
                continue
            subplan_summaries.append(
                {
                    "objective": str(subplan.get("objective", ""))[:220],
                    "tool_summary": self._summarize_tool_results(
                        self._extract_results_from_bundle(subplan.get("tool_execution") or {})
                    )[:900],
                    "continuity_summary": self._summarize_tool_results(
                        self._extract_results_from_bundle(subplan.get("continuity_execution") or {})
                    )[:900],
                }
            )

        return {
            "prompt_preview": str(prompt or "")[:220],
            "plan_metadata": self._plan_metadata_payload(
                plan_text,
                steps,
                response_criteria,
                plan_output,
            ),
            "primary_tool_findings": self._summarize_tool_results(primary_results)[:2400],
            "continuity_tool_findings": self._summarize_tool_results(continuity_results)[:1800],
            "subplan_findings": subplan_summaries,
            "tool_failure_signals": [
                self._serialize_compact(signal, max_chars=320)
                for signal in (tool_failure_signals or [])[:8]
            ],
            "citation_links": self._collect_citation_links(tool_context),
            "image_path_count": len(self._collect_image_paths(tool_context)),
            "context_memory_summaries": [
                str(item.get("summary", ""))[:300]
                for item in (tool_context.get("context_memory") or [])[-5:]
                if isinstance(item, dict)
            ],
        }

    def _context_size(self, entries: list[dict[str, Any]]) -> int:
        return len(self._serialize_compact(entries, max_chars=1_000_000))

    def _compress_context_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if not entries:
            return entries, None

        should_check = len(entries) % self.compression_every == 0
        if not should_check and self._context_size(entries) <= self.max_context_chars:
            return entries, None

        original_size = self._context_size(entries)
        compressed: list[dict[str, Any]] = []

        recent_start = max(0, len(entries) - self.compression_keep_recent)
        for idx, entry in enumerate(entries):
            if idx >= recent_start:
                compressed.append(entry)
                continue
            summary = entry.get("summary") or self._serialize_compact(entry.get("details"), 500)
            compressed.append(
                {
                    "label": entry.get("label", ""),
                    "summary": summary,
                    "details": {"compressed": True, "summary": summary},
                    "timestamp": entry.get("timestamp"),
                }
            )

        while self._context_size(compressed) > self.max_context_chars and len(compressed) > self.compression_keep_recent:
            compressed.pop(0)

        snapshot = {
            "before_chars": original_size,
            "after_chars": self._context_size(compressed),
            "entries_before": len(entries),
            "entries_after": len(compressed),
        }
        return compressed, snapshot

    def _add_context_entry(
        self,
        context_memory: list[dict[str, Any]],
        label: str,
        details: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        entry = {
            "label": label,
            "summary": self._serialize_compact(details, 900),
            "details": details,
            "timestamp": time.time(),
        }
        context_memory.append(entry)
        return self._compress_context_entries(context_memory)

    def _dedupe_response_parts(self, response_parts: list[str]) -> list[str]:
        deduped: list[str] = []
        for part in response_parts:
            part_text = str(part).strip()
            if not part_text:
                continue
            is_duplicate = False
            for existing in deduped:
                similarity = SequenceMatcher(None, existing.lower(), part_text.lower()).ratio()
                if similarity > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(part_text)
        return deduped

    def _build_specific_part_outline(self, part: str, prior_responses: list[str]) -> str:
        part_text = str(part).strip().replace("\n", " ")
        if not prior_responses:
            return f"Part to write:\n{part_text}\n\nWrite only new content for this specific part."

        prior_excerpt = "\n".join(
            f"- {resp.strip().replace(chr(10), ' ')[:220]}"
            for resp in prior_responses[-2:]
        )
        return (
            f"Part to write:\n{part_text}\n\n"
            f"Already covered in previous parts (do not repeat these ideas):\n{prior_excerpt}\n\n"
            "Write only new content for this specific part."
        )

    def _compact_response(self, response: str) -> str:
        paragraphs = [paragraph.strip() for paragraph in response.split("\n\n") if paragraph.strip()]
        compacted: list[str] = []
        for paragraph in paragraphs:
            duplicate = False
            for existing in compacted:
                similarity = SequenceMatcher(None, existing.lower(), paragraph.lower()).ratio()
                if similarity > 0.92:
                    duplicate = True
                    break
            if not duplicate:
                compacted.append(paragraph)
        return "\n\n".join(compacted)

    def _step_keywords(self, step: str) -> list[str]:
        return [word for word in re.findall(r"[a-zA-Z0-9]+", step.lower()) if len(word) >= 4]

    def _step_is_covered(self, response_lower: str, step: str) -> bool:
        keywords = self._step_keywords(step)
        if not keywords:
            return True
        matched = sum(1 for keyword in keywords if keyword in response_lower)
        return matched >= max(1, len(keywords) // 2)

    def _step_coverage_diagnostics(self, response: str, steps: list[str]) -> dict[str, Any]:
        response_lower = (response or "").lower()
        coverage: list[dict[str, Any]] = []
        for step in steps or []:
            step_text = str(step)
            covered = self._step_is_covered(response_lower, step_text)
            coverage.append(
                {
                    "step": step_text,
                    "covered": covered,
                    "keywords": self._step_keywords(step_text),
                }
            )

        covered_count = sum(1 for item in coverage if item["covered"])
        total = len(coverage)
        return {
            "covered_count": covered_count,
            "total_steps": total,
            "coverage_ratio": (covered_count / total) if total else 1.0,
            "missing_steps": [item["step"] for item in coverage if not item["covered"]],
            "per_step": coverage,
        }

    def _ensure_step_coverage(self, response: str, steps: list[str]) -> str:
        if not isinstance(response, str):
            return ""
        if response.strip() or not isinstance(steps, list) or not steps:
            return response
        return "[Graceful degradation] The pipeline could not synthesize a complete answer from the available context."

    def _ensure_inline_citation_markers(self, response: str, citations: dict[str, str]) -> str:
        if not citations:
            return response

        if re.search(r"\[(\d+)\]\(https?://", response) or re.search(r"\[(\d+)\]", response):
            return response

        ordered = sorted(
            citations.keys(),
            key=lambda key: int(key) if str(key).isdigit() else str(key),
        )
        markers = " ".join(f"[{key}]" for key in ordered[:4])
        return response.rstrip() + f"\n\nCitations used: {markers}"

    def _should_avoid_python_tool(self, objective: str, plan_text: str) -> tuple[bool, str]:
        text = f"{objective or ''} {plan_text or ''}".lower()
        compute_markers = (
            "calculate",
            "compute",
            "simulation",
            "simulate",
            "optimiz",
            "equation",
            "matrix",
            "regression",
            "statistic",
            "numerical",
            "algorithm",
            "modeling",
            "dataframe",
            "plot",
            "chart",
            "graph",
        )
        quote_or_citation_markers = (
            "quote",
            "citation",
            "cite",
            "source",
            "reference",
            "url",
            "fact check",
            "verify",
            "page number",
            "summarize",
            "overview",
            "explain",
            "compare viewpoints",
        )
        compute_like = any(token in text for token in compute_markers)
        citation_like = any(token in text for token in quote_or_citation_markers)
        if citation_like and not compute_like:
            return True, "python_route_guard_non_computational"
        return False, ""

    def _citation_fallback_tool_id(self) -> str | None:
        if "web_search_tool_block" in self._tool_lookup:
            return "web_search_tool_block"
        if "wikipedia_search_tool_block" in self._tool_lookup:
            return "wikipedia_search_tool_block"
        return None

    def _suppress_unbacked_citation_markers(self, response: str, citations: dict[str, str]) -> str:
        if not response:
            return response

        def _replace(match: re.Match[str]) -> str:
            key = str(match.group(1))
            return match.group(0) if key in citations else ""

        cleaned = re.sub(r"\[(\d+)\](?!\()", _replace, response)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _suppress_unbacked_image_markers(
        self,
        response: str,
        image_embeddings: list[dict[str, Any]],
    ) -> str:
        if not response:
            return response
        if image_embeddings:
            return response
        cleaned = re.sub(r"\[image_\d+\]", "", response, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _append_page_number_unavailable_note(
        self,
        prompt: str,
        response: str,
    ) -> str:
        prompt_lower = (prompt or "").lower()
        requests_page_numbers = any(
            token in prompt_lower
            for token in ("page number", "page numbers", "which page", "pages for", "page citation")
        )
        if not requests_page_numbers:
            return response

        if re.search(r"\bpage\s+\d+\b|\bp\.\s*\d+\b", response or "", flags=re.IGNORECASE):
            return response

        note = "Page numbers are unavailable in the provided sources."
        if note.lower() in (response or "").lower():
            return response
        return str(response).rstrip() + f"\n\n{note}"

    def _is_math_prompt(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(
            token in lowered
            for token in (
                "solve",
                "equation",
                "math",
                "coefficient",
                "derive",
                "calculate",
                "series",
                "recurrence",
            )
        )

    def _extract_tool_math_answer_candidates(self, tool_context: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for output in self._iter_tool_outputs(tool_context):
            if isinstance(output, dict):
                for key in ("results", "output", "summary", "cited_summary"):
                    value = output.get(key)
                    if not isinstance(value, str):
                        continue
                    lowered = value.lower()
                    if "result" in lowered or "answer" in lowered or "coefficient" in lowered:
                        for match in re.finditer(r"-?\d+(?:\.\d+)?", value, flags=re.IGNORECASE):
                            candidates.append(match.group(0).strip())
        return candidates

    def _extract_response_answer_candidates(self, response: str) -> list[str]:
        return [
            match.group(0).strip()
            for match in re.finditer(r"-?\d+(?:\.\d+)?", response, flags=re.IGNORECASE)
        ]

    def _coherent_math_response(self, prompt: str, response: str, tool_context: dict[str, Any]) -> str:
        if not self._is_math_prompt(prompt):
            return response

        tool_candidates = self._extract_tool_math_answer_candidates(tool_context)
        response_candidates = self._extract_response_answer_candidates(response)
        if not tool_candidates:
            return response

        if response_candidates and any(candidate in tool_candidates for candidate in response_candidates):
            return response

        canonical = tool_candidates[-1]
        return response.rstrip() + f"\n\nFinal answer: {canonical}."

    def _normalize_route_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    def _resolve_tool_id(self, requested_id: str) -> tuple[str | None, str]:
        if requested_id in self._tool_lookup:
            return requested_id, "exact_match"

        requested_normalized = self._normalize_route_token(requested_id)
        for tool_id, tool in self._tool_lookup.items():
            if self._normalize_route_token(tool_id) == requested_normalized:
                return tool_id, "normalized_id_match"
            tool_name = str(tool.details.get("name", ""))
            tool_name_norm = self._normalize_route_token(tool_name)
            if tool_name_norm == requested_normalized:
                return tool_id, "normalized_name_match"
            if requested_normalized and requested_normalized in tool_name_norm:
                return tool_id, "name_contains_match"

        return None, "not_in_lookup"

    def _default_tool_inputs(self, tool_id: str, objective: str, plan_text: str) -> dict[str, Any]:
        objective_text = str(objective or "").strip() or str(plan_text or "").strip()
        objective_lower = objective_text.lower()

        if tool_id == "web_search_tool_block":
            return {"query": objective_text}
        if tool_id == "wikipedia_search_tool_block":
            return {"query": objective_text}
        if tool_id == "python_code_execution_tool_block":
            visuals_needed = any(
                token in objective_lower
                for token in ("plot", "graph", "chart", "visual", "diagram", "figure")
            )
            return {
                "objective": objective_text,
                "visuals_needed": visuals_needed,
            }
        if tool_id == "creative_idea_generator_tool_block":
            return {"objective": objective_text}
        if tool_id == "deductive_reasoning_premise_tool_block":
            return {"objective": objective_text}
        return {"objective": objective_text}

    def _augment_routes_with_tool_hints(
        self,
        routes: list[dict[str, Any]],
        hinted_tool_uses: list[Any],
        objective: str,
        plan_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        route_list = [dict(item) for item in routes if isinstance(item, dict)]
        existing = {
            str(item.get("id"))
            for item in route_list
            if isinstance(item, dict) and item.get("id")
        }
        added: list[dict[str, Any]] = []

        # Backfill missing route inputs so a valid route id without inputs can still execute.
        for route in route_list:
            route_id = route.get("id")
            route_inputs = route.get("inputs")
            if not route_id:
                continue
            if not isinstance(route_inputs, dict) or not route_inputs:
                route["inputs"] = self._default_tool_inputs(
                    tool_id=str(route_id),
                    objective=objective,
                    plan_text=plan_text,
                )
                added.append(
                    {
                        "hint": "router_missing_inputs",
                        "resolved_id": str(route_id),
                        "reason": "default_inputs_backfill",
                    }
                )

        for hint in hinted_tool_uses or []:
            resolved_id, reason = self._resolve_tool_id(str(hint))
            if not resolved_id:
                continue
            if resolved_id in existing:
                continue

            route = {
                "id": resolved_id,
                "inputs": self._default_tool_inputs(
                    tool_id=resolved_id,
                    objective=objective,
                    plan_text=plan_text,
                ),
            }
            route_list.append(route)
            existing.add(resolved_id)
            added.append(
                {
                    "hint": str(hint),
                    "resolved_id": resolved_id,
                    "reason": reason,
                }
            )

        return route_list, added

    def _execute_routes(
        self,
        routes: list[dict[str, Any]],
        *,
        objective: str = "",
        plan_text: str = "",
    ) -> dict[str, Any]:
        execution: dict[str, Any] = {
            "routes_requested": len(routes),
            "routes_executed": 0,
            "skipped_routes": [],
            "results": [],
        }

        for route in routes:
            requested_id = route.get("id")
            route_inputs = route.get("inputs")
            inputs_backfilled = False
            if requested_id and (not isinstance(route_inputs, dict) or not route_inputs):
                route_inputs = self._default_tool_inputs(
                    tool_id=str(requested_id),
                    objective=objective,
                    plan_text=plan_text,
                )
                inputs_backfilled = True

            if not requested_id or not isinstance(route_inputs, dict):
                execution["skipped_routes"].append(
                    {
                        "requested_id": requested_id,
                        "reason": "invalid_inputs",
                    }
                )
                continue

            resolved_id, reason = self._resolve_tool_id(str(requested_id))
            if not resolved_id:
                execution["skipped_routes"].append(
                    {
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "reason": reason,
                    }
                )
                logger.warning(f"Skipping unresolved route id: {requested_id} ({reason})")
                continue

            route_guard_applied = False
            route_guard_reason = ""
            if str(resolved_id) == "python_code_execution_tool_block":
                avoid_python, avoid_reason = self._should_avoid_python_tool(objective, plan_text)
                if avoid_python:
                    fallback_tool_id = self._citation_fallback_tool_id()
                    execution["skipped_routes"].append(
                        {
                            "requested_id": requested_id,
                            "resolved_id": resolved_id,
                            "reason": avoid_reason,
                            "fallback_tool_id": fallback_tool_id,
                        }
                    )
                    if not fallback_tool_id:
                        logger.warning(
                            "Route guard skipped python route with no citation fallback tool available"
                        )
                        continue
                    logger.info(
                        "Route guard replaced python tool with %s for objective '%s...'",
                        fallback_tool_id,
                        str(objective)[:70],
                    )
                    resolved_id = fallback_tool_id
                    route_inputs = self._default_tool_inputs(
                        tool_id=resolved_id,
                        objective=objective,
                        plan_text=plan_text,
                    )
                    route_guard_applied = True
                    route_guard_reason = avoid_reason

            tool = self._tool_lookup[resolved_id]
            effective_inputs = dict(route_inputs)
            if (
                resolved_id == "python_code_execution_tool_block"
                and not self.allow_visual_outputs
            ):
                effective_inputs["visuals_needed"] = False
            try:
                output = tool.process(effective_inputs)
                if (
                    resolved_id == "python_code_execution_tool_block"
                    and not self.allow_visual_outputs
                    and isinstance(output, dict)
                ):
                    output["plots"] = []
                execution["results"].append(
                    {
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "inputs": effective_inputs,
                        "inputs_backfilled": inputs_backfilled,
                        "route_guard_applied": route_guard_applied,
                        "route_guard_reason": route_guard_reason,
                        "output": output,
                        "error": None,
                    }
                )
                execution["routes_executed"] += 1
            except Exception as exc:
                error_message = str(exc)
                logger.warning(f"Tool execution failed for {resolved_id}: {error_message}")
                execution["results"].append(
                    {
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "inputs": effective_inputs,
                        "inputs_backfilled": inputs_backfilled,
                        "route_guard_applied": route_guard_applied,
                        "route_guard_reason": route_guard_reason,
                        "output": None,
                        "error": error_message,
                    }
                )
                execution["skipped_routes"].append(
                    {
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "reason": "tool_error",
                        "error": error_message,
                    }
                )

        return execution

    def _extract_results_from_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        by_tool: dict[str, Any] = {}
        for item in bundle.get("results", []):
            if not isinstance(item, dict):
                continue
            resolved_id = item.get("resolved_id")
            if not resolved_id:
                continue
            by_tool[str(resolved_id)] = item.get("output")
        return by_tool

    def _looks_like_failure_text(self, text: str) -> bool:
        lowered = (text or "").lower()
        failure_tokens = (
            "failed",
            "failure",
            "error",
            "exception",
            "traceback",
            "could not",
            "unable to",
            "timed out",
        )
        return any(token in lowered for token in failure_tokens)

    def _collect_failure_signals_from_bundle(
        self,
        bundle_name: str,
        bundle: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(bundle, dict):
            return []

        signals: list[dict[str, Any]] = []
        for skipped in bundle.get("skipped_routes", []):
            if not isinstance(skipped, dict):
                continue
            signals.append(
                {
                    "bundle": bundle_name,
                    "kind": "skipped_route",
                    "requested_id": skipped.get("requested_id"),
                    "resolved_id": skipped.get("resolved_id"),
                    "reason": skipped.get("reason"),
                    "error": skipped.get("error"),
                }
            )

        for result in bundle.get("results", []):
            if not isinstance(result, dict):
                continue
            requested_id = result.get("requested_id")
            resolved_id = result.get("resolved_id")
            route_inputs = result.get("inputs", {})
            output = result.get("output")
            route_error = result.get("error")

            if route_error:
                signals.append(
                    {
                        "bundle": bundle_name,
                        "kind": "route_error",
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "error": route_error,
                    }
                )

            if isinstance(output, dict):
                text_candidates = [
                    output.get("results"),
                    output.get("summary"),
                    output.get("cited_summary"),
                    output.get("conclusion_reasoning"),
                    output.get("conclusion"),
                ]
                for candidate in text_candidates:
                    if isinstance(candidate, str) and self._looks_like_failure_text(candidate):
                        signals.append(
                            {
                                "bundle": bundle_name,
                                "kind": "tool_reported_failure_text",
                                "requested_id": requested_id,
                                "resolved_id": resolved_id,
                                "snippet": candidate[:300],
                            }
                        )
                        break

                if str(resolved_id) == "python_code_execution_tool_block":
                    visuals_needed = bool((route_inputs or {}).get("visuals_needed"))
                    plots = output.get("plots")
                    if visuals_needed and isinstance(plots, list) and not plots:
                        signals.append(
                            {
                                "bundle": bundle_name,
                                "kind": "missing_expected_plot_output",
                                "requested_id": requested_id,
                                "resolved_id": resolved_id,
                                "visuals_needed": visuals_needed,
                            }
                        )
            elif isinstance(output, str) and self._looks_like_failure_text(output):
                signals.append(
                    {
                        "bundle": bundle_name,
                        "kind": "tool_reported_failure_text",
                        "requested_id": requested_id,
                        "resolved_id": resolved_id,
                        "snippet": output[:300],
                    }
                )

        return signals

    def _collect_tool_failure_signals(self, tool_context: dict[str, Any]) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        signals.extend(
            self._collect_failure_signals_from_bundle(
                "primary_execution",
                tool_context.get("primary_execution"),
            )
        )
        signals.extend(
            self._collect_failure_signals_from_bundle(
                "primary_continuity_execution",
                tool_context.get("primary_continuity_execution"),
            )
        )

        for idx, subplan in enumerate(tool_context.get("subplans", []), start=1):
            if not isinstance(subplan, dict):
                continue
            signals.extend(
                self._collect_failure_signals_from_bundle(
                    f"subplan_{idx}_execution",
                    subplan.get("tool_execution"),
                )
            )
            signals.extend(
                self._collect_failure_signals_from_bundle(
                    f"subplan_{idx}_continuity_execution",
                    subplan.get("continuity_execution"),
                )
            )

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for signal in signals:
            signature = self._serialize_compact(signal, max_chars=400)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(signal)
        return deduped

    def _iter_tool_outputs(self, tool_context: dict[str, Any]) -> Iterator[Any]:
        for output in self._extract_results_from_bundle(tool_context.get("primary_execution") or {}).values():
            yield output

        for subplan in tool_context.get("subplans", []):
            for output in self._extract_results_from_bundle(subplan.get("tool_execution") or {}).values():
                yield output
            for output in self._extract_results_from_bundle(subplan.get("continuity_execution") or {}).values():
                yield output

    def _collect_citation_links(self, tool_context: dict[str, Any]) -> dict[str, str]:
        citations: dict[str, str] = {}
        next_index = 1

        for output in self._iter_tool_outputs(tool_context):
            if not isinstance(output, dict):
                continue

            links = output.get("search_result_links")
            if isinstance(links, dict):
                for key, value in links.items():
                    url = str(value).strip()
                    if not url:
                        continue
                    citations[str(key)] = url

            wiki_url = output.get("url")
            if isinstance(wiki_url, str) and wiki_url.strip():
                while str(next_index) in citations:
                    next_index += 1
                citations[str(next_index)] = wiki_url.strip()
                next_index += 1

        return citations

    def _collect_image_paths(self, tool_context: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for output in self._iter_tool_outputs(tool_context):
            if not isinstance(output, dict):
                continue
            plots = output.get("plots")
            if not isinstance(plots, list):
                continue
            for plot_path in plots:
                plot_str = str(plot_path)
                if plot_str in seen:
                    continue
                seen.add(plot_str)
                paths.append(plot_str)
        return paths

    def _path_to_data_uri(
        self,
        image_path: str,
        max_bytes: int = 6_000_000,
    ) -> tuple[str | None, str | None]:
        path = Path(str(image_path))
        if not path.exists() or not path.is_file():
            return None, "file_not_found"

        try:
            size = path.stat().st_size
            if size <= 0:
                return None, "file_empty"
            if size > max_bytes:
                return None, f"file_too_large_{size}_bytes"

            payload = path.read_bytes()
            mime_type, _ = mimetypes.guess_type(path.name)
            if not mime_type:
                mime_type = "image/png"
            encoded = base64.b64encode(payload).decode("ascii")
            return f"data:{mime_type};base64,{encoded}", None
        except Exception as exc:
            return None, str(exc)

    def _collect_image_embeddings(
        self,
        tool_context: dict[str, Any],
        max_images: int = 6,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.enable_image_embedding:
            return [], []

        embeddings: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []

        for index, image_path in enumerate(self._collect_image_paths(tool_context), start=1):
            if len(embeddings) >= max_images:
                issues.append(
                    {
                        "path": image_path,
                        "reason": "max_images_limit_reached",
                    }
                )
                continue

            data_uri, error = self._path_to_data_uri(image_path)
            if error or not data_uri:
                issues.append({"path": image_path, "reason": error or "unknown"})
                continue

            embeddings.append(
                {
                    "index": index,
                    "path": image_path,
                    "media_type": data_uri.split(";", 1)[0].replace("data:", ""),
                    "data_uri": data_uri,
                }
            )

        return embeddings, issues

    def _embed_citations(self, response: str, citations: dict[str, str]) -> str:
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return f"[{key}]"

        return re.sub(r"\[(\d+)\]\(https?://[^\)]+\)", _replace, response)

    def _linkify_citation_markers(self, response: str, citations: dict[str, str]) -> str:
        if not citations:
            return response

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            url = citations.get(key)
            if not url:
                return match.group(0)
            return f"[{key}]({url})"

        # Replace plain numeric markers only when not already markdown links.
        return re.sub(r"\[(\d+)\](?!\()", _replace, response)

    def _image_markdown(self, image_index: int, data_uri: str) -> str:
        return f"![image_{image_index}]({data_uri})"

    def _embed_images_naturally(
        self,
        response: str,
        image_embeddings: list[dict[str, Any]],
    ) -> str:
        updated = str(response or "").rstrip()
        if not image_embeddings:
            return updated

        # Remove legacy path/token image blocks so only markdown embeds remain.
        updated = re.sub(
            r"(?is)\n\nImages:\n(?:-?\s*(?:/[^ \n]+|image_\d+|\[image_\d+\])\s*\n?)+\s*$",
            "",
            updated,
        ).rstrip()

        placed_indexes: set[int] = set()
        for idx, item in enumerate(image_embeddings, start=1):
            data_uri = str(item.get("data_uri", "")).strip()
            if not data_uri:
                continue

            marker = self._image_markdown(idx, data_uri)
            # Prefer explicit placeholder replacement when present.
            candidate_patterns = [
                re.compile(rf"\[image_{idx}\]", flags=re.IGNORECASE),
                re.compile(rf"\bimage_{idx}\b", flags=re.IGNORECASE),
            ]
            replaced = False
            for pattern in candidate_patterns:
                updated, count = pattern.subn(marker, updated, count=1)
                if count > 0:
                    placed_indexes.add(idx)
                    replaced = True
                    break

            if replaced:
                continue

        remaining_markers: list[str] = []
        for idx, item in enumerate(image_embeddings, start=1):
            if idx in placed_indexes:
                continue
            data_uri = str(item.get("data_uri", "")).strip()
            if not data_uri:
                continue
            remaining_markers.append(self._image_markdown(idx, data_uri))

        if not remaining_markers:
            return updated

        marker_block = "\n".join(remaining_markers)
        paragraphs = [segment for segment in re.split(r"\n{2,}", updated) if segment.strip()]
        if not paragraphs:
            return marker_block

        anchor_pattern = re.compile(
            r"\b(plot|graph|chart|figure|visual|image|diagram|parabola)\b",
            flags=re.IGNORECASE,
        )
        anchor_index = next(
            (index for index, paragraph in enumerate(paragraphs) if anchor_pattern.search(paragraph)),
            0,
        )
        paragraphs.insert(anchor_index + 1, marker_block)
        return "\n\n".join(paragraphs)

    def _append_reference_and_image_sections(
        self,
        response: str,
        citations: dict[str, str],
        image_embeddings: list[dict[str, Any]],
    ) -> str:
        updated = self._embed_images_naturally(response, image_embeddings)
        has_sources_section = bool(re.search(r"(?im)^sources:\s*$", updated))

        if citations and not has_sources_section:
            ordered_keys = sorted(
                set(re.findall(r"\[(\d+)\]", updated)),
                key=lambda value: int(value),
            )
            if not ordered_keys:
                ordered_keys = sorted(citations.keys(), key=lambda value: int(value) if value.isdigit() else value)

            sources_lines = [f"- [{key}]({citations[key]})" for key in ordered_keys if key in citations]
            if sources_lines:
                updated += "\n\nSources:\n" + "\n".join(sources_lines)

        return updated

    def _enrich_response(
        self,
        response: str,
        tool_context: dict[str, Any],
        image_embeddings: list[dict[str, Any]],
    ) -> str:
        citations = self._collect_citation_links(tool_context)
        with_inline = self._embed_citations(response, citations)
        with_inline = self._linkify_citation_markers(with_inline, citations)
        with_inline = self._suppress_unbacked_citation_markers(with_inline, citations)
        with_inline = self._suppress_unbacked_image_markers(with_inline, image_embeddings)
        return self._append_reference_and_image_sections(with_inline, citations, image_embeddings)

    def _coverage_count(self, response: str, steps: list[str]) -> int:
        response_lower = (response or "").lower()
        return sum(
            1 for step in steps or [] if self._step_is_covered(response_lower, str(step))
        )

    def _mentions_failure(self, response: str) -> bool:
        return self._looks_like_failure_text(response or "")

    def _select_synthesis_candidate(
        self,
        original: str,
        candidate: str,
        steps: list[str],
        tool_failure_signals: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        original_text = str(original or "").strip()
        candidate_text = str(candidate or "").strip()
        diagnostics: dict[str, Any] = {
            "selected": "original",
            "reasons": [],
            "original_score": 0.0,
            "candidate_score": 0.0,
        }

        if not candidate_text:
            diagnostics["reasons"].append("candidate_empty")
            return original_text, diagnostics

        candidate_lower = candidate_text.lower()
        if candidate_lower.startswith(("improved plan", "revised plan")):
            diagnostics["reasons"].append("candidate_looks_like_plan_not_answer")
            return original_text, diagnostics

        if "[graceful degradation]" in candidate_lower and "[graceful degradation]" not in original_text.lower():
            diagnostics["reasons"].append("candidate_introduced_degraded_marker")
            return original_text, diagnostics

        original_coverage = self._coverage_count(original_text, steps)
        candidate_coverage = self._coverage_count(candidate_text, steps)

        original_score = float(original_coverage)
        candidate_score = float(candidate_coverage)

        if candidate_coverage < original_coverage:
            candidate_score -= 2.0
            diagnostics["reasons"].append("candidate_lost_step_coverage")

        has_failures = bool(tool_failure_signals)
        if has_failures:
            if self._mentions_failure(original_text):
                original_score += 1.0
            if not self._mentions_failure(candidate_text):
                candidate_score -= 1.5
                diagnostics["reasons"].append("candidate_hides_tool_failures")

        if len(candidate_text) < max(60, int(len(original_text) * 0.5)):
            candidate_score -= 1.0
            diagnostics["reasons"].append("candidate_too_short_vs_original")

        if any(token in candidate_lower for token in ("steps_for_improvement", "target_schema", "critique:")):
            candidate_score -= 1.0
            diagnostics["reasons"].append("candidate_contains_internal_review_artifacts")

        diagnostics["original_score"] = original_score
        diagnostics["candidate_score"] = candidate_score

        if candidate_score >= original_score:
            diagnostics["selected"] = "candidate"
            return candidate_text, diagnostics
        return original_text, diagnostics

    def _contains_degraded_marker(self, value: Any) -> bool:
        if isinstance(value, str):
            return "[Graceful degradation]" in value
        if isinstance(value, dict):
            return any(self._contains_degraded_marker(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_degraded_marker(item) for item in value)
        return False

    def _summarize_tool_results(self, tool_results: dict[str, Any]) -> str:
        lines: list[str] = []
        for tool_id, result in tool_results.items():
            if isinstance(result, dict):
                compact = ", ".join(sorted(result.keys()))
                lines.append(f"{tool_id}: keys={compact}")
            else:
                lines.append(f"{tool_id}: {str(result)[:240]}")
        return "\n".join(lines)

    def _estimate_eta(self, start_time: float, completed_units: int, projected_units: int) -> int | None:
        if completed_units <= 0 or projected_units <= completed_units:
            return None
        elapsed = time.time() - start_time
        average_per_unit = elapsed / max(completed_units, 1)
        remaining = projected_units - completed_units
        return max(0, int(average_per_unit * remaining))

    def _emit_event(
        self,
        run_id: str,
        event_type: str,
        stage: str,
        payload: dict[str, Any],
        start_time: float,
        completed_units: int,
        projected_units: int,
    ) -> dict[str, Any]:
        event = {
            "run_id": run_id,
            "timestamp": time.time(),
            "event_type": event_type,
            "stage": stage,
            "eta_seconds": self._estimate_eta(start_time, completed_units, projected_units),
            "payload": payload,
        }
        self.event_history.append(event)
        return event

    def run_stream(
        self,
        prompt: str,
        thinking_level: ThinkingLevel = "med-synth",
    ) -> Iterator[dict[str, Any]]:
        run_id = str(uuid.uuid4())
        start_time = time.time()
        completed_units = 0
        projected_units = 120

        self.event_history = []
        level = self._normalize_thinking_level(thinking_level)

        tool_context: dict[str, Any] = {
            "primary_routing": None,
            "primary_execution": None,
            "primary_continuity_routing": None,
            "primary_continuity_execution": None,
            "subplans": [],
            "context_memory": [],
            "compression_snapshots": [],
            "long_response_parts": [],
            "plan_review": {},
            "synthesis_review": {},
        }
        degraded_notes: list[str] = []

        available_tools = self._available_tools()

        yield self._emit_event(
            run_id,
            "run_started",
            "init",
            {
                "prompt_preview": prompt[:120],
                "thinking_level": level,
                "tools_enabled": [tool.details["id"] for tool in self.tools],
                "settings": {
                    "plan_review_enabled": self._plan_review_enabled(level),
                    "synthesis_review_enabled": self._synth_review_enabled(level),
                },
            },
            start_time,
            completed_units,
            projected_units,
        )

        logger.info(
            f"Pipeline.run_stream() start - prompt: {prompt[:50]}..., thinking_level: {level}"
        )

        try:
            completed_units += 10
            yield self._emit_event(
                run_id,
                "stage_started",
                "initial_plan",
                {"message": "Generating initial plan"},
                start_time,
                completed_units,
                projected_units,
            )

            planner = InitialPlanCreationBlock()
            try:
                plan_output = planner.process(
                    {
                        "prompt": prompt,
                        "available_tools": available_tools,
                    }
                )
                plan_output = self._normalize_audience_fields(plan_output)
            except Exception as exc:
                degraded_notes.append(f"Planner failed: {exc}")
                logger.warning(f"Planner failed: {exc}")
                plan_output = {
                    "general_plan": "[Graceful degradation] Initial plan failed; proceeding with minimal response path.",
                    "steps": [prompt],
                    "complex_response": False,
                    "long_response": False,
                    "response_criteria": [],
                    "assumed_audience": "Unknown",
                    "assumed_audience_knowledge_level": "Unknown",
                    "assumed_audience_reading_level": "Unknown",
                }

            plan_text = str(plan_output.get("general_plan", "")).strip()
            steps = [str(step).strip() for step in plan_output.get("steps", []) if str(step).strip()]
            complex_response = bool(plan_output.get("complex_response"))
            long_response = bool(plan_output.get("long_response"))
            response_criteria = [
                str(item).strip()
                for item in plan_output.get("response_criteria", [])
                if str(item).strip()
            ]
            response_criteria = self._ensure_naturalism_criterion(response_criteria)
            plan_output["response_criteria"] = response_criteria

            if level == "low":
                complex_response = False
                yield self._emit_event(
                    run_id,
                    "warning",
                    "post_low_mode_policy",
                    {
                        "message": "Low thinking level forced non-complex routing only (long responses still allowed)",
                        "source": "low_mode_policy",
                    },
                    start_time,
                    completed_units,
                    projected_units,
                )

            yield self._emit_event(
                run_id,
                "variables_snapshot",
                "planning",
                self._planning_snapshot_payload(
                    complex_response,
                    long_response,
                    steps,
                    response_criteria,
                    plan_output,
                ),
                start_time,
                completed_units,
                projected_units,
            )

            plan_review_data: dict[str, Any] = {}
            if self._plan_review_enabled(level):
                completed_units += 8
                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "plan_review",
                    {"message": "Running self-critique and improvement on plan"},
                    start_time,
                    completed_units,
                    projected_units,
                )

                critique_block = SelfCritiqueBlock()
                improvement_block = ImprovementBlock()

                try:
                    critique = critique_block.process(
                        {
                            "item": plan_text,
                            "objective": prompt,
                            "context": {
                                "prompt": prompt,
                                "plan": plan_output,
                            },
                        }
                    )

                    improved = improvement_block.process(
                        {
                            "item": json.dumps(plan_output, ensure_ascii=True),
                            "critique": str(critique.get("general_critique", "")),
                            "objective": prompt,
                            "target_schema": "plan",
                        }
                    )
                    improved_item = improved.get("improved_item", {})
                    if isinstance(improved_item, dict):
                        improved_item.update(
                            {
                                "assumed_audience": improved_item.get(
                                    "assumed_audience", "General audience"
                                ),
                                "assumed_audience_knowledge_level": improved_item.get(
                                    "assumed_audience_knowledge_level", "Mixed"
                                ),
                                "assumed_audience_reading_level": improved_item.get(
                                    "assumed_audience_reading_level", "General"
                                ),
                            }
                        )
                        plan_output = self._normalize_audience_fields(improved_item)
                        plan_text = str(plan_output.get("general_plan", plan_text)).strip()
                        steps = [
                            str(step).strip()
                            for step in plan_output.get("steps", steps)
                            if str(step).strip()
                        ]
                        complex_response = bool(plan_output.get("complex_response", complex_response))
                        long_response = bool(plan_output.get("long_response", long_response))
                        response_criteria = [
                            str(item).strip()
                            for item in plan_output.get("response_criteria", response_criteria)
                            if str(item).strip()
                        ]
                        response_criteria = self._ensure_naturalism_criterion(response_criteria)
                        plan_output["response_criteria"] = response_criteria

                    plan_review_data = {
                        "critique": critique,
                        "improvement": improved,
                        "post_plan": plan_output,
                    }
                    tool_context["plan_review"] = plan_review_data
                except Exception as exc:
                    degraded_notes.append(f"Plan review failed: {exc}")
                    yield self._emit_event(
                        run_id,
                        "warning",
                        "post_plan_review",
                        {"message": f"Plan review failed: {exc}", "source": "plan_review"},
                        start_time,
                        completed_units,
                        projected_units,
                    )

            subplans: list[dict[str, Any]] = []

            if complex_response and steps:
                completed_units += 4
                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "subplans",
                    {
                        "message": "Complex response detected, creating subplans",
                        "total_subplans": len(steps),
                    },
                    start_time,
                    completed_units,
                    projected_units,
                )

                subplan_block = SubPlanCreationBlock()
                router = PrimaryToolRouterBlock()
                sub_router = SubToolRouterBlock()
                rolling_subplan_context: list[dict[str, Any]] = []

                for i, step in enumerate(steps, start=1):
                    completed_units += 2
                    yield self._emit_event(
                        run_id,
                        "subplan_started",
                        "subplans",
                        {
                            "index": i,
                            "total": len(steps),
                            "step": step,
                            "prior_subplans": rolling_subplan_context,
                        },
                        start_time,
                        completed_units,
                        projected_units,
                    )

                    context_payload = {
                        "subplan_index": i,
                        "total_subplans": len(steps),
                        "prior_subplans": rolling_subplan_context[-3:],
                        "compressed_context": [
                            item.get("summary") for item in tool_context["context_memory"][-5:]
                        ],
                    }

                    try:
                        subplan_output = subplan_block.process(
                            {
                                "plan": plan_text,
                                "objective": step,
                                "context": context_payload,
                            }
                        )
                    except Exception as exc:
                        degraded_notes.append(f"Subplan {i} creation failed: {exc}")
                        subplan_output = {
                            "sub_plan": f"[Graceful degradation] Subplan generation failed for step: {step}",
                            "steps": [step],
                            "tool_uses": [],
                        }

                    subplan_text = str(subplan_output.get("sub_plan", step)).strip()

                    try:
                        routing = router.process(
                            {
                                "plan": subplan_text,
                                "objective": step,
                                "available_tools": available_tools,
                            }
                        )
                        routes = routing.get("routes", [])
                        routes, hinted_added = self._augment_routes_with_tool_hints(
                            routes=routes,
                            hinted_tool_uses=subplan_output.get("tool_uses", []),
                            objective=step,
                            plan_text=subplan_text,
                        )
                        routing["routes"] = routes
                        if hinted_added:
                            routing["hint_backfill"] = hinted_added
                        continuity = bool(routing.get("continuity"))
                    except Exception as exc:
                        degraded_notes.append(f"Subplan {i} routing failed: {exc}")
                        routes = []
                        continuity = False
                        routing = {
                            "routes": [],
                            "continuity": False,
                            "error": str(exc),
                        }

                    tool_execution = self._execute_routes(
                        routes,
                        objective=step,
                        plan_text=subplan_text,
                    )
                    tool_results = self._extract_results_from_bundle(tool_execution)

                    continuity_execution = None
                    continuity_routing = None
                    continuity_results: dict[str, Any] = {}
                    if continuity:
                        try:
                            continuity_routing = sub_router.process(
                                {
                                    "tool_output": self._summarize_tool_results(tool_results),
                                    "objective": step,
                                    "plan": {
                                        "main_plan": plan_text,
                                        "subplan": subplan_text,
                                    },
                                    "available_tools": available_tools,
                                }
                            )
                            continuity_execution = self._execute_routes(
                                continuity_routing.get("routes", []),
                                objective=step,
                                plan_text=subplan_text,
                            )
                            continuity_results = self._extract_results_from_bundle(
                                continuity_execution
                            )
                        except Exception as exc:
                            degraded_notes.append(f"Subplan {i} continuity routing failed: {exc}")
                            continuity_execution = {
                                "routes_requested": 0,
                                "routes_executed": 0,
                                "skipped_routes": [],
                                "results": [],
                                "error": str(exc),
                            }

                    summary_entry = {
                        "step": step,
                        "sub_plan": subplan_text,
                        "tool_findings": self._summarize_tool_results(
                            {
                                **tool_results,
                                **continuity_results,
                            }
                        )[:300],
                    }
                    rolling_subplan_context.append(summary_entry)

                    subplan_record = {
                        "step": step,
                        "subplan": subplan_output,
                        "context_used": context_payload,
                        "routes": routing,
                        "tool_execution": tool_execution,
                        "tool_results": tool_results,
                        "continuity_routes": continuity_routing,
                        "continuity_execution": continuity_execution,
                        "continuity_results": continuity_results,
                    }
                    subplans.append(subplan_record)
                    tool_context["subplans"].append(subplan_record)

                    tool_context["context_memory"], compression_snapshot = self._add_context_entry(
                        tool_context["context_memory"],
                        f"subplan_{i}",
                        {
                            "step": step,
                            "sub_plan": subplan_text,
                            "tool_findings": self._summarize_tool_results(
                                {
                                    **tool_results,
                                    **continuity_results,
                                }
                            ),
                        },
                    )
                    if compression_snapshot:
                        tool_context["compression_snapshots"].append(compression_snapshot)

                    completed_units += 2
                    yield self._emit_event(
                        run_id,
                        "subplan_completed",
                        "subplans",
                        {
                            "index": i,
                            "total": len(steps),
                            "step": step,
                            "routes_requested": tool_execution.get("routes_requested", 0),
                            "routes_executed": tool_execution.get("routes_executed", 0),
                            "context_carried_entries": len(rolling_subplan_context),
                            "subplan_record": subplan_record,
                        },
                        start_time,
                        completed_units,
                        projected_units,
                    )
            else:
                completed_units += 6
                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "primary_routing",
                    {"message": "Using primary routing path"},
                    start_time,
                    completed_units,
                    projected_units,
                )

                try:
                    primary_routing = PrimaryToolRouterBlock().process(
                        {
                            "plan": plan_text,
                            "objective": prompt,
                            "available_tools": available_tools,
                        }
                    )
                    primary_routes, hinted_added = self._augment_routes_with_tool_hints(
                        routes=primary_routing.get("routes", []),
                        hinted_tool_uses=plan_output.get("tool_uses", []),
                        objective=prompt,
                        plan_text=plan_text,
                    )
                    primary_routing["routes"] = primary_routes
                    if hinted_added:
                        primary_routing["hint_backfill"] = hinted_added
                except Exception as exc:
                    degraded_notes.append(f"Primary routing failed: {exc}")
                    primary_routing = {
                        "routes": [],
                        "continuity": False,
                        "error": str(exc),
                    }

                primary_execution = self._execute_routes(
                    primary_routing.get("routes", []),
                    objective=prompt,
                    plan_text=plan_text,
                )
                primary_results = self._extract_results_from_bundle(primary_execution)

                continuity_routing = None
                continuity_execution = None
                continuity_results: dict[str, Any] = {}
                if bool(primary_routing.get("continuity")):
                    try:
                        continuity_routing = SubToolRouterBlock().process(
                            {
                                "tool_output": self._summarize_tool_results(primary_results),
                                "objective": prompt,
                                "plan": {
                                    "main_plan": plan_text,
                                    "steps": steps,
                                },
                                "available_tools": available_tools,
                            }
                        )
                        continuity_execution = self._execute_routes(
                            continuity_routing.get("routes", []),
                            objective=prompt,
                            plan_text=plan_text,
                        )
                        continuity_results = self._extract_results_from_bundle(
                            continuity_execution
                        )
                    except Exception as exc:
                        degraded_notes.append(f"Continuity routing failed: {exc}")
                        continuity_execution = {
                            "routes_requested": 0,
                            "routes_executed": 0,
                            "skipped_routes": [],
                            "results": [],
                            "error": str(exc),
                        }

                tool_context["primary_routing"] = primary_routing
                tool_context["primary_execution"] = primary_execution
                tool_context["primary_continuity_routing"] = continuity_routing
                tool_context["primary_continuity_execution"] = continuity_execution

                tool_context["context_memory"], compression_snapshot = self._add_context_entry(
                    tool_context["context_memory"],
                    "primary_routing",
                    {
                        "plan": plan_text,
                        "tool_findings": self._summarize_tool_results(
                            {
                                **primary_results,
                                **continuity_results,
                            }
                        ),
                    },
                )
                if compression_snapshot:
                    tool_context["compression_snapshots"].append(compression_snapshot)

            tool_failure_signals = self._collect_tool_failure_signals(tool_context)
            if tool_failure_signals:
                degraded_notes.append(
                    f"Detected {len(tool_failure_signals)} tool failure signal(s) during execution."
                )

            response_parts: list[str] = []
            raw_synthesis_response = ""
            synthesis_review_data: dict[str, Any] = {}
            synthesis_selection: dict[str, Any] = {}

            if long_response:
                completed_units += 8
                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "long_synthesis",
                    {"message": "Breaking long response into parts"},
                    start_time,
                    completed_units,
                    projected_units,
                )

                long_router = LargeResponseRouterBlock()
                try:
                    long_routing = long_router.process(
                        {
                            "plan": self._plan_metadata_payload(
                                plan_text,
                                steps,
                                response_criteria,
                                plan_output,
                            ),
                            "objective": prompt,
                            "response_criteria": response_criteria,
                            "tool_context": tool_context,
                        }
                    )
                    response_parts = self._dedupe_response_parts(
                        long_routing.get("response_parts", [])
                    )
                except Exception as exc:
                    degraded_notes.append(f"Long response router failed: {exc}")
                    response_parts = ["[Graceful degradation] single fallback synthesis part"]

                if not response_parts:
                    response_parts = ["Core response to the user prompt"]

                synth = LongResponseSynthesisBlock()
                responses: list[str] = []

                for idx, part in enumerate(response_parts, start=1):
                    completed_units += 2
                    yield self._emit_event(
                        run_id,
                        "synthesis_part_started",
                        "long_synthesis",
                        {
                            "index": idx,
                            "total": len(response_parts),
                            "part": part,
                        },
                        start_time,
                        completed_units,
                        projected_units,
                    )

                    outline = self._build_specific_part_outline(part, responses)
                    try:
                        part_output = synth.process(
                            {
                                "tool_context": tool_context,
                                "prompt": {"prompt": prompt},
                                "plan": self._plan_metadata_payload(
                                    plan_text,
                                    steps,
                                    response_criteria,
                                    plan_output,
                                ),
                                "specific_part_outline": outline,
                            }
                        )
                        part_text = str(part_output.get("synthesis", "")).strip()
                        if not part_text:
                            part_text = "[Graceful degradation] Failed to synthesize this section."
                            degraded_notes.append(f"Synthesis part {idx} returned empty content")
                    except Exception as exc:
                        degraded_notes.append(f"Synthesis part {idx} failed: {exc}")
                        part_text = "[Graceful degradation] Failed to synthesize this section."

                    responses.append(part_text)
                    tool_context["long_response_parts"].append(
                        {
                            "index": idx,
                            "part": part,
                            "outline": outline,
                            "response": part_text,
                        }
                    )

                    yield self._emit_event(
                        run_id,
                        "synthesis_part_completed",
                        "long_synthesis",
                        {
                            "index": idx,
                            "total": len(response_parts),
                            "response": part_text,
                        },
                        start_time,
                        completed_units,
                        projected_units,
                    )

                raw_synthesis_response = self._compact_response("\n\n".join(responses))

                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "synthesis_review",
                    {
                        "message": "Skipping synthesis critique/improvement for long-response mode",
                        "skipped": True,
                    },
                    start_time,
                    completed_units,
                    projected_units,
                )
            else:
                completed_units += 8
                yield self._emit_event(
                    run_id,
                    "stage_started",
                    "synthesis",
                    {"message": "Generating standard synthesis"},
                    start_time,
                    completed_units,
                    projected_units,
                )

                try:
                    synthesis_output = SynthesisBlock().process(
                        {
                            "tool_context": tool_context,
                            "prompt": {"prompt": prompt},
                            "plan": self._plan_metadata_payload(
                                plan_text,
                                steps,
                                response_criteria,
                                plan_output,
                            ),
                        }
                    )
                    raw_synthesis_response = str(synthesis_output.get("synthesis", "")).strip()
                except Exception as exc:
                    degraded_notes.append(f"Synthesis failed: {exc}")
                    raw_synthesis_response = "[Graceful degradation] Synthesis failed; returning fallback response."

                if self._synth_review_enabled(level):
                    completed_units += 6
                    yield self._emit_event(
                        run_id,
                        "stage_started",
                        "synthesis_review",
                        {"message": "Running synthesis self-critique and improvement"},
                        start_time,
                        completed_units,
                        projected_units,
                    )

                    try:
                        critique = SelfCritiqueBlock().process(
                            {
                                "item": raw_synthesis_response,
                                "objective": prompt,
                                "context": self._build_synthesis_critique_context(
                                    prompt=prompt,
                                    plan_text=plan_text,
                                    steps=steps,
                                    response_criteria=response_criteria,
                                    plan_output=plan_output,
                                    tool_context=tool_context,
                                    tool_failure_signals=tool_failure_signals,
                                ),
                            }
                        )
                        improved = ImprovementBlock().process(
                            {
                                "item": raw_synthesis_response,
                                "critique": str(critique.get("general_critique", "")),
                                "objective": prompt,
                                "target_schema": "synthesis",
                            }
                        )

                        improved_item = improved.get("improved_item", {})
                        if isinstance(improved_item, dict):
                            candidate_synthesis = str(
                                improved_item.get("synthesis", raw_synthesis_response)
                            ).strip()
                            raw_synthesis_response, synthesis_selection = (
                                self._select_synthesis_candidate(
                                    original=raw_synthesis_response,
                                    candidate=candidate_synthesis,
                                    steps=steps,
                                    tool_failure_signals=tool_failure_signals,
                                )
                            )

                        synthesis_review_data = {
                            "critique": critique,
                            "improvement": improved,
                            "selection": synthesis_selection,
                        }
                        tool_context["synthesis_review"] = synthesis_review_data
                    except Exception as exc:
                        degraded_notes.append(f"Synthesis review failed: {exc}")

            tool_context["synthesis_selection"] = synthesis_selection

            assembly_trace: dict[str, Any] = {
                "raw_synthesis_response": raw_synthesis_response,
            }
            final_response = raw_synthesis_response
            final_response = self._coherent_math_response(prompt, final_response, tool_context)
            assembly_trace["after_math_coherence"] = final_response

            step_coverage = self._step_coverage_diagnostics(final_response, steps)
            final_response = self._ensure_step_coverage(final_response, steps)
            assembly_trace["after_step_coverage_guard"] = final_response
            assembly_trace["step_coverage"] = step_coverage

            citation_links = self._collect_citation_links(tool_context)
            final_response = self._ensure_inline_citation_markers(final_response, citation_links)
            assembly_trace["after_inline_citations"] = final_response
            image_embeddings, image_embedding_issues = self._collect_image_embeddings(tool_context)
            final_response = self._enrich_response(final_response, tool_context, image_embeddings)
            assembly_trace["after_reference_enrichment"] = final_response
            final_response = self._append_page_number_unavailable_note(prompt, final_response)
            assembly_trace["after_page_note"] = final_response

            image_paths = self._collect_image_paths(tool_context)
            if image_embedding_issues:
                degraded_notes.append(
                    f"{len(image_embedding_issues)} image(s) could not be embedded."
                )
            degraded_mode_active = bool(degraded_notes) or bool(tool_failure_signals) or self._contains_degraded_marker(
                {
                    "tool_context": tool_context,
                    "response": final_response,
                }
            )

            rca_bundle = {
                "prompt": prompt,
                "thinking_level": level,
                "plan": plan_output,
                "steps": steps,
                "complex_response": complex_response,
                "long_response": long_response,
                "response_criteria": response_criteria,
                "plan_review": plan_review_data,
                "synthesis_review": synthesis_review_data,
                "synthesis_selection": synthesis_selection,
                "tool_context": tool_context,
                "tool_failure_signals": tool_failure_signals,
                "degraded_notes": degraded_notes,
                "assembly_trace": assembly_trace,
                "image_embeddings": image_embeddings,
                "image_embedding_issues": image_embedding_issues,
            }

            result = {
                "plan": plan_output,
                "steps": steps,
                "complex_response": complex_response,
                "long_response": long_response,
                "response_criteria": response_criteria,
                "plan_review": plan_review_data,
                "synthesis_review": synthesis_review_data,
                "subplans": subplans,
                "tool_context": tool_context,
                "context_memory": tool_context.get("context_memory", []),
                "compression_snapshots": tool_context.get("compression_snapshots", []),
                "response_parts": response_parts,
                "citation_links": citation_links,
                "image_paths": image_paths,
                "image_embeddings": image_embeddings,
                "image_embedding_issues": image_embedding_issues,
                "tool_failure_signals": tool_failure_signals,
                "degraded_mode_active": degraded_mode_active,
                "degraded_notes": degraded_notes,
                "step_coverage": step_coverage,
                "synthesis_selection": synthesis_selection,
                "assembly_trace": assembly_trace,
                "rca_bundle": rca_bundle,
                "raw_synthesis_response": raw_synthesis_response,
                "response": final_response,
            }

            completed_units += 10
            yield self._emit_event(
                run_id,
                "final_result",
                "complete",
                result,
                start_time,
                completed_units,
                projected_units,
            )

            completed_units = projected_units
            yield self._emit_event(
                run_id,
                "complete",
                "complete",
                {
                    "duration_seconds": round(time.time() - start_time, 2),
                    "event_count": len(self.event_history),
                    "degraded_mode_active": degraded_mode_active,
                },
                start_time,
                completed_units,
                projected_units,
            )

        except TimeoutError:
            raise
        except Exception as exc:
            logger.error(f"Fatal pipeline error: {exc}")
            yield self._emit_event(
                run_id,
                "error",
                "error",
                {
                    "message": str(exc),
                    "degraded_notes": degraded_notes,
                },
                start_time,
                completed_units,
                projected_units,
            )

    def run(
        self,
        prompt: str,
        thinking_level: ThinkingLevel = "med-synth",
        include_events: bool = True,
    ) -> dict[str, Any]:
        final_result: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []

        for event in self.run_stream(prompt=prompt, thinking_level=thinking_level):
            events.append(event)
            if event.get("event_type") == "final_result":
                final_result = event.get("payload")

        if final_result is None:
            final_result = {
                "plan": "",
                "steps": [],
                "complex_response": False,
                "long_response": False,
                "response_criteria": [],
                "tool_context": {},
                "citation_links": {},
                "image_paths": [],
                "image_embeddings": [],
                "image_embedding_issues": [],
                "tool_failure_signals": [],
                "degraded_mode_active": True,
                "degraded_notes": ["Pipeline ended without final_result event"],
                "step_coverage": {
                    "covered_count": 0,
                    "total_steps": 0,
                    "coverage_ratio": 0.0,
                    "missing_steps": [],
                    "per_step": [],
                },
                "synthesis_selection": {},
                "assembly_trace": {},
                "rca_bundle": {},
                "response": "[Graceful degradation] No final result available.",
            }

        if include_events:
            final_result["events"] = events
            final_result["event_count"] = len(events)
        return final_result


class PipelineBuilder:
    def __init__(self, tools: list[Any] | Literal["all"] = "all"):
        if tools == "all":
            tool_classes = _get_all_tool_classes()
            selected_tools: list[Any] = []
            for tool_class in tool_classes:
                try:
                    selected_tools.append(tool_class())
                except Exception:
                    continue
            self.tools = selected_tools
        elif isinstance(tools, list):
            self.tools = tools
        else:
            self.tools = []

    def build(self) -> Pipeline:
        return Pipeline(tools=self.tools)
