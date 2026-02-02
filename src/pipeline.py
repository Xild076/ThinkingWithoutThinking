import inspect
import re
from typing import Any, Generator, Literal
import time

from src.pipeline_blocks import ToolBlock, PlannerPromptBlock, SelfCritiqueBlock, ImprovementCritiqueBlock, ToolRouterBlock, PythonExecutionToolBlock, WebSearchToolBlock, CreativeIdeaGeneration, ResponseSynthesizerBlock
from src.logger import ExecutionTrace
TOOL_MAX_RETRIES = 3
TOOL_RETRY_DELAY = 1.0


class Pipeline:
    def __init__(self, tools: list[ToolBlock] | None = None):
        self.tools = tools if tools else [WebSearchToolBlock(), PythonExecutionToolBlock(), CreativeIdeaGeneration()]
        self.trace = ExecutionTrace()
        self.tool_errors: list[dict[str, Any]] = []

    def _invoke_tool_with_retry(self, tool: ToolBlock, inputs: dict[str, Any]) -> dict[str, Any]:
        last_error = None
        for attempt in range(1, TOOL_MAX_RETRIES + 1):
            try:
                result = tool(**inputs)
                return {"success": True, "result": result, "attempts": attempt}
            except Exception as e:
                last_error = str(e)
                self.trace.log("tool_retry", f"Tool {tool.identity} attempt {attempt} failed: {last_error}")
                if attempt < TOOL_MAX_RETRIES:
                    time.sleep(TOOL_RETRY_DELAY * attempt)
        
        return {"success": False, "error": last_error, "attempts": TOOL_MAX_RETRIES}

    def _invoke_tool(self, tool: ToolBlock, inputs: dict[str, Any]):
        return tool(**inputs)

    def stream(self, prompt: str, prompt_overrides: dict[str, str] = None, thinking_level: Literal["low", "medium", "high"] = "medium") -> Generator[dict, None, None]:
        self.trace = ExecutionTrace()
        self.trace.log("init", "Pipeline started", {"thinking_level": thinking_level, "prompt_length": len(prompt)})
        
        if prompt_overrides is None:
            prompt_overrides = {}
        
        planner = PlannerPromptBlock()
        self_critique = SelfCritiqueBlock()
        improvement_critique = ImprovementCritiqueBlock()
        tool_router = ToolRouterBlock()
        response_synthesizer = ResponseSynthesizerBlock()
        
        if "planner_prompt_block" in prompt_overrides:
            planner.prompt = prompt_overrides["planner_prompt_block"]
        if "self_critique_block" in prompt_overrides:
            self_critique.prompt = prompt_overrides["self_critique_block"]
        if "improvement_critique_block" in prompt_overrides:
            improvement_critique.prompt = prompt_overrides["improvement_critique_block"]
        if "tool_router_block" in prompt_overrides:
            tool_router.prompt = prompt_overrides["tool_router_block"]
        if "response_synthesizer_block" in prompt_overrides:
            response_synthesizer.prompt = prompt_overrides["response_synthesizer_block"]

        try:
            yield {"type": "planning", "message": "Generating plan..."}
            self.trace.log("planning", "Starting plan generation")
            plan = planner(prompt)
            self.trace.log("planning", "Plan generated", {"plan_length": len(str(plan))})
            self.trace.log("planner_io", "Planner input/output", {"input": prompt, "output": plan})
            yield {"type": "plan", "plan": plan}
            self.trace.log_trace("plan", {"plan": plan})

            init_task = prompt
            plan_context = f"USER PROMPT:\\n{prompt}\\n\\nPLAN:\\n{plan}"

            if thinking_level == "high":
                yield {"type": "critique", "message": "Critiquing plan..."}
                self.trace.log("critique", "Starting plan critique")
                critique = self_critique(prompt, plan, init_task)
                self.trace.log("critique", "Plan critiqued", {"critique_length": len(str(critique))})
                self.trace.log("self_critique_io", "Self-critique input/output", {"input": prompt, "output": plan, "initial_task": init_task, "critique": critique})
                yield {"type": "plan_critique", "critique": critique}
                self.trace.log_trace("plan_critique", {"critique": critique})
                
                yield {"type": "improvement", "message": "Improving plan..."}
                self.trace.log("improvement", "Starting plan improvement")
                plan = improvement_critique(prompt, plan, critique, init_task)
                self.trace.log("improvement", "Plan improved", {"new_plan_length": len(str(plan))})
                self.trace.log("improvement_io", "Improvement input/output", {"input": prompt, "output": plan, "critique": critique, "initial_task": init_task})
                yield {"type": "plan_improved", "plan": plan}
                self.trace.log_trace("plan_improved", {"plan": plan})

            yield {"type": "routing", "message": "Routing to tools..."}
            self.trace.log("routing", "Starting tool routing")
            routed = tool_router(prompt, plan, self.tools)
            
            if not routed:
                self.trace.log("routing", "No tools selected")
                yield {"type": "warning", "message": "No tools selected for execution"}
            else:
                self.trace.log("routing", f"Tools selected: {len(routed)}", {"tools": [r['tool'].identity for r in routed]})
            
            routed_serializable = [{"tool_id": item["tool"].identity, "tool_description": item["tool"].details.get("description", ""), "inputs": item["inputs"]} for item in routed]
            yield {"type": "routed", "tools": routed_serializable}
            self.trace.log_trace("routed", {"tools": routed_serializable})
            self.trace.log("tool_router_io", "Tool router input/output", {"input": {"prompt": prompt, "plan": plan}, "output": routed_serializable})

            tool_outputs = {}
            self.tool_errors = []
            
            for item in routed:
                tool = item.get("tool")
                inputs = item.get("inputs", {})
                if not tool:
                    continue

                if any(v == "" for v in inputs.values()):
                    self.trace.log("tool_invoke", f"Skipped tool with missing inputs: {tool.identity}", {"inputs": inputs})
                    yield {"type": "warning", "message": f"Skipped {tool.identity} due to missing inputs"}
                    continue
                
                self.trace.log("tool_invoke", f"Starting tool: {tool.identity}", {"inputs_keys": list(inputs.keys())})
                yield {"type": "tool_start", "tool_id": tool.identity, "description": tool.details.get("description", "")}
                self.trace.log_trace("tool_start", {"tool_id": tool.identity, "inputs": inputs})
                
                invoke_result = self._invoke_tool_with_retry(tool, inputs)
                
                if invoke_result["success"]:
                    result = invoke_result["result"]
                    self.trace.log("tool_invoke", f"Tool succeeded: {tool.identity}", 
                                   {"result_size": len(str(result)), "attempts": invoke_result["attempts"]})
                else:
                    error_msg = invoke_result["error"]
                    result = {"error": error_msg}
                    self.tool_errors.append({
                        "tool_id": tool.identity,
                        "error": error_msg,
                        "attempts": invoke_result["attempts"]
                    })
                    self.trace.log_error("tool_invoke", f"Tool failed after {invoke_result['attempts']} attempts: {tool.identity}", 
                                         {"error": error_msg})
                    yield {"type": "tool_error", "tool_id": tool.identity, 
                           "error": f"Tool failed after {invoke_result['attempts']} retries: {error_msg}"}
                
                tool_outputs[tool.identity] = result
                yield {"type": "tool_complete", "tool_id": tool.identity, "result": result}
                self.trace.log("tool_io", "Tool input/output", {"tool_id": tool.identity, "input": inputs, "output": result})
                self.trace.log_trace("tool_complete", {"tool_id": tool.identity, "result": result})

            self.trace.log("synthesis", "Starting synthesis", {"num_tool_results": len(tool_outputs)})
            yield {"type": "synthesizing", "message": "Synthesizing response..."}
            
            successful_outputs = {}
            for tool_id, result in tool_outputs.items():
                if isinstance(result, dict):
                    if tool_id == "web_search_tool_block":
                        if result.get("summarize") or result.get("links"):
                            successful_outputs[tool_id] = result
                    elif not result.get("error"):
                        successful_outputs[tool_id] = result
                elif result:
                    successful_outputs[tool_id] = result

            all_plots_base64 = []
            for result in tool_outputs.values():
                if isinstance(result, dict) and result.get("plots_base64"):
                    all_plots_base64.extend(result["plots_base64"])

            image_tokens = []
            image_markdown = {}
            if all_plots_base64:
                for idx, b64 in enumerate(all_plots_base64, start=1):
                    token = f"[[image_{idx}]]"
                    image_tokens.append(token)
                    image_markdown[token] = f"![Plot {idx}](data:image/png;base64,{b64})"
            
            evidence_sources = []
            if not successful_outputs:
                tool_outputs_with_names = (
                    "[NO EXTERNAL SOURCES AVAILABLE]\n"
                    "No external data was gathered. The response must be based on general knowledge only.\n"
                    "Do NOT cite any URLs, publications, or specific statistics."
                )
            else:
                evidence_parts = []
                for tool_id, result in successful_outputs.items():
                    if tool_id == "web_search_tool_block" and isinstance(result, dict):
                        summary = result.get("summary") or result.get("summarize", "")
                        sources = result.get("sources", [])
                        if sources:
                            source_lines = []
                            for src in sources:
                                src_id = src.get("id") or f"S{len(source_lines) + 1}"
                                url = src.get("url", "")
                                snippet = src.get("snippet", "")
                                source_lines.append(f"[{src_id}] {url}\n  - {snippet}")
                                evidence_sources.append({"id": src_id, "url": url})
                        else:
                            links = result.get("links", [])
                            source_lines = [f"[S{idx + 1}] {link}" for idx, link in enumerate(links)] if links else ["[no sources returned]"]
                            for idx, link in enumerate(links):
                                evidence_sources.append({"id": f"S{idx + 1}", "url": link})
                        evidence_parts.append(
                            "=== Source: web_search_tool_block ===\n"
                            f"SUMMARY:\n{summary}\nSOURCES:\n" + "\n".join(source_lines)
                        )
                        continue

                    cleaned = result
                    if isinstance(result, dict):
                        cleaned = {k: v for k, v in result.items() if k not in ("plots_base64", "visuals")}
                    text = str(cleaned)
                    if len(text) > 800:
                        text = f"{text[:800]}..."
                    evidence_parts.append(f"=== Source: {tool_id} ===\n{text}")

                tool_outputs_with_names = "\n\n".join(evidence_parts)

            if image_tokens:
                image_context = "\n".join(f"{token} = Plot {idx + 1}" for idx, token in enumerate(image_tokens))
                tool_outputs_with_names = f"{tool_outputs_with_names}\n\nIMAGES AVAILABLE:\n{image_context}\nUse these image tokens exactly in the response where images should appear."
            self.trace.log_trace("evidence", {"evidence": tool_outputs_with_names})
            
            response_payload = response_synthesizer(prompt, {"combined": tool_outputs_with_names}, plan)
            response_body = ""
            used_sources = []
            if isinstance(response_payload, dict):
                response_body = response_payload.get("body", "")
                used_sources = response_payload.get("used_sources") or []
            else:
                response_body = str(response_payload)

            if successful_outputs and "general knowledge" in response_body.lower():
                lines = [line for line in response_body.splitlines() if "general knowledge" not in line.lower()]
                response_body = "\n".join(lines).strip()

            if successful_outputs:
                response_body = re.sub(r'https?://\\S+', '', response_body).strip()
            self.trace.log("synthesizer_io", "Synthesizer input/output", {"input": {"prompt": prompt, "plan": plan, "sources": tool_outputs_with_names}, "output": response_body})
            
            from src.prompt_validator import PromptValidator
            is_citation_valid, citation_issues = PromptValidator.validate_response_citations(
                response_body, tool_outputs_with_names
            )
            
            if not is_citation_valid:
                self.trace.log_error(
                    "synthesis", 
                    f"Detected {len(citation_issues)} citation issues",
                    {"issues": citation_issues}
                )
                response_body = PromptValidator.strip_hallucinated_citations(response_body)
                yield {"type": "warning", "message": "Response modified: removed unverified citations"}
            
            self.trace.log("synthesis", "Response synthesized", {"response_length": len(str(response_body))})
            yield {"type": "response", "response": response_body}
            self.trace.log_trace("response", {"response": response_body})

            if thinking_level in ("medium", "high"):
                self.trace.log("final_review", "Starting final critique")
                yield {"type": "final_critique", "message": "Final critique..."}
                final_critique = self_critique(prompt, response_body, plan_context)
                self.trace.log("final_review", "Final critique complete")
                self.trace.log("final_self_critique_io", "Final self-critique input/output", {"input": prompt, "output": response_body, "initial_task": plan_context, "critique": final_critique})
                yield {"type": "final_critique_complete", "critique": final_critique}
                
                yield {"type": "final_improvement", "message": "Final refinement..."}
                self.trace.log("final_review", "Starting final improvement")
                final_response_body = improvement_critique(prompt, response_body, final_critique, plan_context)
                self.trace.log("final_review", "Final improvement complete")
                self.trace.log("final_improvement_io", "Final improvement input/output", {"input": prompt, "output": final_response_body, "critique": final_critique, "initial_task": plan_context})
            else:
                final_response_body = response_body

            sources_section = ""
            if evidence_sources:
                used = [s for s in evidence_sources if not used_sources or s["id"] in used_sources]
                if not used:
                    used = evidence_sources
                sources_section = "Sources:\n" + "\n".join(f"[{s['id']}] {s['url']}" for s in used)

            final_response = final_response_body.strip()
            if sources_section:
                final_response = f"{final_response}\n\n{sources_section}"

            if image_markdown:
                for token, md in image_markdown.items():
                    final_response = final_response.replace(token, md)

            yield {"type": "final_response", "response": final_response, "plots_base64": all_plots_base64}
            self.trace.log_trace("final_response", {"response": final_response, "plots_base64": all_plots_base64})

            trace_summary = self.trace.get_summary()
            self.trace.log("complete", "Pipeline completed", {"total_errors": trace_summary["total_errors"]})
            
            final_output = {
                "plan": plan, 
                "tools": routed_serializable, 
                "tool_outputs": tool_outputs, 
                "response": final_response,
                "plots_base64": all_plots_base64,
                "trace": trace_summary,
                "tool_errors": self.tool_errors
            }
            yield {"type": "complete", "final": final_output}
            complete_payload = dict(final_output)
            complete_payload["trace"] = {
                "total_events": trace_summary.get("total_events"),
                "total_errors": trace_summary.get("total_errors"),
                "duration": trace_summary.get("duration")
            }
            self.trace.log_trace("complete", {"final": complete_payload})
        
        except Exception as e:
            self.trace.log_error("pipeline", f"Fatal error: {str(e)}")
            yield {"type": "error", "message": f"Pipeline error: {str(e)}", "trace": self.trace.get_summary()}
