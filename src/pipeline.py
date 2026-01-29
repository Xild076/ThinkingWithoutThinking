import inspect
from typing import Any, Generator, Literal

from src.pipeline_blocks import ToolBlock, PlannerPromptBlock, SelfCritiqueBlock, ImprovementCritiqueBlock, ToolRouterBlock, PythonExecutionToolBlock, WebSearchToolBlock, CreativeIdeaGeneration, ResponseSynthesizerBlock
from src.logger import ExecutionTrace
from src.parameter_mapper import ParameterMapper


class Pipeline:
    def __init__(self, tools: list[ToolBlock] | None = None):
        self.tools = tools if tools else [WebSearchToolBlock(), PythonExecutionToolBlock(), CreativeIdeaGeneration()]
        self.trace = ExecutionTrace()

    def _invoke_tool(self, tool: ToolBlock, inputs: dict[str, Any]):
        return tool(**inputs)

    def stream(self, prompt: str, prompt_overrides: dict[str, str] = None, thinking_level: Literal["low", "medium_synth", "medium_plan", "high"] = "medium_synth") -> Generator[dict, None, None]:
        self.trace = ExecutionTrace()
        self.trace.log("init", "Pipeline started", {"thinking_level": thinking_level, "prompt_length": len(prompt)})
        
        if prompt_overrides is None:
            prompt_overrides = {}
        
        # Initialize blocks
        planner = PlannerPromptBlock()
        self_critique = SelfCritiqueBlock()
        improvement_critique = ImprovementCritiqueBlock()
        tool_router = ToolRouterBlock()
        response_synthesizer = ResponseSynthesizerBlock()
        
        # Apply prompt overrides if they exist
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
            plan = planner(prompt) if thinking_level != "low" else prompt
            self.trace.log("planning", "Plan generated", {"plan_length": len(str(plan))})
            yield {"type": "plan", "plan": plan}

            init_task = prompt

            if thinking_level in ("medium_plan", "high"):
                yield {"type": "critique", "message": "Critiquing plan..."}
                self.trace.log("critique", "Starting plan critique")
                critique = self_critique(prompt, plan, init_task)
                self.trace.log("critique", "Plan critiqued", {"critique_length": len(str(critique))})
                yield {"type": "plan_critique", "critique": critique}
                
                yield {"type": "improvement", "message": "Improving plan..."}
                self.trace.log("improvement", "Starting plan improvement")
                plan = improvement_critique(prompt, plan, critique, init_task)
                self.trace.log("improvement", "Plan improved", {"new_plan_length": len(str(plan))})
                yield {"type": "plan_improved", "plan": plan}

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

            tool_outputs = {}
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
                
                try:
                    result = self._invoke_tool(tool, inputs)
                    self.trace.log("tool_invoke", f"Tool succeeded: {tool.identity}", {"result_size": len(str(result))})
                except Exception as exc:
                    error_msg = str(exc)
                    result = {"error": error_msg}
                    self.trace.log_error("tool_invoke", f"Tool failed: {tool.identity}", {"error": error_msg})
                
                tool_outputs[tool.identity] = result
                yield {"type": "tool_complete", "tool_id": tool.identity, "result": result}

            self.trace.log("synthesis", "Starting synthesis", {"num_tool_results": len(tool_outputs)})
            yield {"type": "synthesizing", "message": "Synthesizing response..."}
            
            tool_outputs_with_names = "\n\n".join(
                f"From {tool_id}:\n{str(result)[:500]}{'...' if len(str(result)) > 500 else ''}"
                for tool_id, result in tool_outputs.items()
            )
            response = response_synthesizer(prompt, {"combined": tool_outputs_with_names}, plan)
            self.trace.log("synthesis", "Response synthesized", {"response_length": len(str(response))})
            yield {"type": "response", "response": response}

            if thinking_level in ("medium_synth", "high"):
                self.trace.log("final_review", "Starting final critique")
                yield {"type": "final_critique", "message": "Final critique..."}
                final_critique = self_critique(prompt, response, init_task)
                self.trace.log("final_review", "Final critique complete")
                yield {"type": "final_critique_complete", "critique": final_critique}
                
                yield {"type": "final_improvement", "message": "Final refinement..."}
                self.trace.log("final_review", "Starting final improvement")
                final_response = improvement_critique(prompt, response, final_critique, init_task)
                self.trace.log("final_review", "Final improvement complete")
                yield {"type": "final_response", "response": final_response}
            else:
                final_response = response

            trace_summary = self.trace.get_summary()
            self.trace.log("complete", "Pipeline completed", {"total_errors": trace_summary["total_errors"]})
            yield {"type": "complete", "final": {"plan": plan, "tools": routed_serializable, "tool_outputs": tool_outputs, "response": final_response, "trace": trace_summary}}
        
        except Exception as e:
            self.trace.log_error("pipeline", f"Fatal error: {str(e)}")
            yield {"type": "error", "message": f"Pipeline error: {str(e)}", "trace": self.trace.get_summary()}

