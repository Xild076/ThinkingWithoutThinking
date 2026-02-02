import inspect
import json
import logging
import re
from random import random
from src.utility import generate_text, load_prompts, reload_prompts
from src.tool_utility import python_exec_tool, retrieve_text_many, retrieve_links
from src.parameter_mapper import ParameterMapper

logger = logging.getLogger(__name__)

from enum import Enum

from typing import Any

from pydantic import BaseModel, Field

# Load prompts once at import time for default behavior
_cached_prompts: dict[str, str] | None = None


def get_prompts(force_reload: bool = False) -> dict[str, str]:
    """Get prompts, optionally reloading from disk.
    
    Args:
        force_reload: If True, reload from disk even if cached
        
    Returns:
        Dictionary of prompt_id -> prompt_text
    """
    global _cached_prompts
    if _cached_prompts is None or force_reload:
        _cached_prompts = load_prompts('prompts.json')
    return _cached_prompts


# Initialize cache on import
_cached_prompts = load_prompts('prompts.json')


class CreativityLevel(float, Enum):
    STRICT = 0.0
    LOW = 0.5
    MEDIUM = 1.0
    HIGH = 1.5
    ROUGHLY_RANDOM = 2.0


class PlannerPlanSchema(BaseModel):
    role: str = Field(description="The persona the execution agent should adopt.")
    objective: str = Field(description="The overall objective of the plan.")
    feasibility: str = Field(description="Feasibility assessment.")
    safety: str = Field(description="Safety assessment.")
    tools: list[str] = Field(description="Tool categories needed, e.g. Web Search, Code Execution, None.")
    steps: list[str] = Field(description="Atomic, ordered steps to complete the task.")
    search_queries: list[str] = Field(description="Specific search queries if external info is needed.")
    success_metrics: list[str] = Field(description="Clear success criteria for the final response.")


class SelfCritiqueSchema(BaseModel):
    verdict: str = Field(description="PASS or NEEDS REVISION.")
    critical_issues: list[str] = Field(description="Major flaws that must be fixed.")
    minor_issues: list[str] = Field(description="Minor improvements.")
    praise: list[str] = Field(description="What was done well.")


class ImprovementSchema(BaseModel):
    improved_output: str = Field(description="The revised output.")


class SynthesizerSchema(BaseModel):
    body: str = Field(description="The response body with embedded citations like [S1].")
    used_sources: list[str] = Field(default_factory=list, description="List of source IDs cited in the body.")


class WebSourceSchema(BaseModel):
    id: str = Field(description="Stable identifier like S1, S2.")
    url: str = Field(description="Source URL.")
    snippet: str = Field(description="Short snippet supporting the summary.")


class WebSearchSummarySchema(BaseModel):
    summary: str = Field(description="Concise summary grounded in the sources.")
    sources: list[WebSourceSchema] = Field(description="List of cited sources with IDs.")

class PipelineBlock:
    details: dict = {
        "description": "Base pipeline block.",
        "prompt_creation_criteria": "The criteria to follow for automated prompt creation.",
        "id": "base_pipeline_block",
        "inputs": ["No inputs."],
        "outputs": ["No outputs."]
    }
    
    def __init__(self):
        self.identity = "base_pipeline_block"
        self.prompt = get_prompts().get(self.identity, "")

    def _task_context(self) -> str:
        description = self.details.get("description", "")
        inputs = ", ".join(self.details.get("inputs", []))
        outputs = ", ".join(self.details.get("outputs", []))
        # Provide task-level context to keep the model aligned with the block's role.
        return f"### TASK CONTEXT\nDescription: {description}\nInputs: {inputs}\nOutputs: {outputs}\n"

    def _with_task_context(self, prompt: str) -> str:
        base_prompt = ""
        if self.identity != "base_pipeline_block":
            base_prompt = get_prompts().get("base_pipeline_block", "")
        parts = [p for p in (base_prompt, self._task_context(), prompt) if p]
        return "\n\n".join(parts)
    
    def __call__(self, *args, **kwds):
        raise NotImplementedError("This method should be overridden by subclasses.")


class ToolBlock(PipelineBlock):
    details: dict = {
        "description": "Base tool block.",
        "id": "base_tool_block",
        "inputs": ["No inputs."],
        "outputs": ["No outputs."]
    }
    
    def __init__(self):
        self.identity = "base_tool_block"
    
    def __call__(self, *args, **kwds):
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class PlannerPromptBlock(PipelineBlock):
    details: dict = {
        "description": "Generates a plan based on the given objective.",
        "prompt_creation_criteria": "The prompt must have the system define: a role, the objective, constraints, and evaluation criteria. The prompt may add additional instructions as needed so long as they align with these elements.",
        "id": "planner_prompt_block",
        "inputs": ["prompt (str): The initial user prompt to plan for."],
        "outputs": ["plan (str): The generated plan."]
    }

    def __init__(self):
        self.identity = "planner_prompt_block"
        self.prompt = get_prompts().get(self.identity, "")
    
    def __call__(self, prompt: str) -> str:
        full_prompt = self.prompt.replace("{prompt}", prompt)
        full_prompt = self._with_task_context(full_prompt)
        plan_obj = generate_text(
            prompt=full_prompt,
            model='gemma',
            schema=PlannerPlanSchema,
            temperature=CreativityLevel.MEDIUM.value,
            max_tokens=2048
        )
        plan_lines = [
            f"ROLE: {plan_obj.role}",
            f"OBJECTIVE: {plan_obj.objective}",
            f"FEASIBILITY: {plan_obj.feasibility}",
            f"SAFETY: {plan_obj.safety}",
            f"TOOLS: {', '.join(plan_obj.tools)}",
            "STEPS:"
        ]
        for idx, step in enumerate(plan_obj.steps):
            cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', step or '').strip()
            plan_lines.append(f"{idx + 1}. {cleaned}")
        if plan_obj.search_queries:
            plan_lines.append("SEARCH QUERIES:")
            plan_lines.extend(f"- {q}" for q in plan_obj.search_queries)
        if plan_obj.success_metrics:
            plan_lines.append("SUCCESS METRICS:")
            plan_lines.extend(f"- {m}" for m in plan_obj.success_metrics)
        return "\n".join(plan_lines)


class SelfCritiqueBlock(PipelineBlock):
    details: dict = {
        "description": "Generates a critique based on the given input and output.",
        "prompt_creation_criteria": "The prompt must instruct the system to analyze the input and output with full consideration of the initial task and its area of management in mind, identify any issues or areas for improvement, and provide constructive feedback. This must be generalizable and applicable to a wide range of inputs and outputs. The prompt may add additional instructions as needed so long as they align with these elements.",
        "id": "self_critique_block",
        "inputs": [
            "input (str): The initial prompt inputted.",
            "output (str): The output to be critiqued.",
            "initial_task (str): The initial task or objective."
        ],
        "outputs": ["critique (str): The generated self-critique."]
    }

    def __init__(self):
        self.identity = "self_critique_block"
        self.prompt = get_prompts().get(self.identity, "")
    
    def __call__(self, input: str, output: str, init_task: str) -> str:
        full_prompt = self.prompt.replace("{input}", input).replace("{output}", output).replace("{initial_task}", init_task)
        full_prompt = self._with_task_context(full_prompt)
        critique_obj = generate_text(
            prompt=full_prompt,
            model='nemotron',
            schema=SelfCritiqueSchema,
            temperature=CreativityLevel.LOW.value,
            max_tokens=1024
        )
        critique_lines = [
            f"VERDICT: {critique_obj.verdict}",
            "CRITICAL ISSUES:"
        ]
        critique_lines.extend(f"- {issue}" for issue in (critique_obj.critical_issues or ["None."]))
        critique_lines.append("MINOR ISSUES:")
        critique_lines.extend(f"- {issue}" for issue in (critique_obj.minor_issues or ["None."]))
        critique_lines.append("PRAISE:")
        critique_lines.extend(f"- {item}" for item in (critique_obj.praise or ["None."]))
        return "\n".join(critique_lines)


class ImprovementCritiqueBlock(PipelineBlock):
    details: dict = {
        "description": "Generates improvement suggestions based on the given critique with full consideration of the initial task and its area of management in mind.",
        "id": "improvement_critique_block",
        "inputs": [
            "input (str): The initial prompt inputted.",
            "output (str): The output to be improved.",
            "critique (str): The critique to base improvements on.",
            "initial_task (str): The initial task or objective."
        ],
        "outputs": ["improvements (str): The generated improvement suggestions."]
    }

    def __init__(self):
        self.identity = "improvement_critique_block"
        self.prompt = get_prompts().get(self.identity, "")
    
    def __call__(self, input: str, output: str, critique: str, init_task: str) -> str:
        full_prompt = self.prompt.replace("{input}", input).replace("{output}", output).replace("{critique}", critique).replace("{initial_task}", init_task)
        full_prompt = self._with_task_context(full_prompt)
        improvements_obj = generate_text(
            prompt=full_prompt,
            model='gemma',
            schema=ImprovementSchema,
            temperature=CreativityLevel.MEDIUM.value,
            max_tokens=2048
        )
        return improvements_obj.improved_output


class ToolChoice(BaseModel):
    tool_id: str = Field(description="The identity of the chosen tool (matches ToolBlock.identity or details['id']).")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Inputs to pass when calling the tool.")


class ToolRouterSchema(BaseModel):
    tool_choice: list[ToolChoice] = Field(description="The chosen tool to use along with key parameters.")


class ToolRouterBlock(PipelineBlock):
    details: dict = {
        "description": "Routes to the appropriate tool based on the given prompt and plan.",
        "id": "tool_router_block",
        "prompt_creation_criteria": "The prompt must instruct the system to analyze the given prompt and plan, consider the available tools and their capabilities, and select the most appropriate tool(s) to use. The prompt should guide the system to provide a structured response that includes the chosen tool(s) and the necessary inputs for each tool. This must be generalizable and applicable to a wide range of prompts and plans.",
        "inputs": [
            "prompt (str): The initial user prompt.",
            "plan (str): The generated plan."
        ],
        "outputs": ["tool_choice (list): The chosen tool to use."]
    }

    def __init__(self):
        self.identity = "tool_router_block"
        self.prompt = get_prompts().get(self.identity, "")
    
    def _extract_json_from_response(self, raw: str) -> str:
        """Extract JSON from various response formats."""
        cleaned = raw.strip()
        
        # Remove markdown code fences
        if "```" in cleaned:
            # Find content between code fences
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    cleaned = part
                    break
        
        # Find the JSON object/array
        start_brace = cleaned.find("{")
        start_bracket = cleaned.find("[")
        
        if start_brace == -1 and start_bracket == -1:
            return "{\"tool_choice\": []}"
        
        # Use whichever comes first
        if start_brace == -1:
            start = start_bracket
        elif start_bracket == -1:
            start = start_brace
        else:
            start = min(start_brace, start_bracket)
        
        # Find matching closing bracket
        depth = 0
        end = start
        opening = cleaned[start]
        closing = "}" if opening == "{" else "]"
        
        for i, char in enumerate(cleaned[start:], start):
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        return cleaned[start:end]

    def _parse_tool_choice(self, raw: str) -> list[ToolChoice]:
        """Parse tool choice from LLM response with multiple fallback strategies."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            cleaned = self._extract_json_from_response(raw)
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw: {raw[:200]}")
            return []

        # Handle various response formats
        if isinstance(data, list):
            # Direct list of tool choices
            data = {"tool_choice": data}
        
        if not isinstance(data, dict):
            return []
        
        # Try to find tool_choice in various keys
        tool_choices = data.get("tool_choice") or data.get("tools") or data.get("selected_tools") or []
        
        if not isinstance(tool_choices, list):
            tool_choices = [tool_choices] if tool_choices else []
        
        try:
            return ToolRouterSchema.model_validate({"tool_choice": tool_choices}).tool_choice
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
            return []
    
    def __call__(self, prompt: str, plan: str, allowed_tools: list[ToolBlock]) -> list[dict]:
        import logging
        logger = logging.getLogger(__name__)
        
        # Build tool info with clearer formatting
        possible_tools = []
        for tool in allowed_tools:
            tool_info = (
                f"TOOL: {tool.identity}\n"
                f"  Description: {tool.details['description']}\n"
                f"  Inputs: {', '.join(tool.details['inputs'])}"
            )
            possible_tools.append(tool_info)
        
        full_prompt = self.prompt.replace("{tools}", "\n\n".join(possible_tools)).replace("{prompt}", prompt).replace("{plan}", plan)
        full_prompt = self._with_task_context(full_prompt)
        
        llm_choices = []
        
        # Strategy 1: Try schema-based parsing
        try:
            tool_choice = generate_text(
                prompt=full_prompt,
                model='nemotron',
                schema=ToolRouterSchema,
                temperature=CreativityLevel.STRICT.value,  # Use 0.0 for deterministic routing
            )
            llm_choices = tool_choice.tool_choice
            logger.debug(f"Schema parsing succeeded: {len(llm_choices)} tools")
        except Exception as e:
            logger.warning(f"Schema parsing failed: {e}")
        
        # Strategy 2: If schema failed, try raw text with stricter prompt
        if not llm_choices:
            logger.info("Attempting raw JSON parsing fallback")
            retry_prompt = (
                f"{full_prompt}\n\n"
                "IMPORTANT: Return ONLY a JSON object. No other text.\n"
                "Example: {\"tool_choice\": [{\"tool_id\": \"web_search_tool_block\", \"inputs\": {\"query\": \"search term\"}}]}"
            )
            try:
                raw_choice = generate_text(
                    prompt=retry_prompt,
                    model='nemotron',
                    temperature=CreativityLevel.STRICT.value,
                    max_tokens=512
                )
                llm_choices = self._parse_tool_choice(raw_choice)
                logger.debug(f"Raw parsing result: {len(llm_choices)} tools from: {raw_choice[:100]}")
            except Exception as e:
                logger.warning(f"Raw parsing also failed: {e}")
        
        # Strategy 3: Try with gemma as backup model
        if not llm_choices and self._plan_suggests_tools(plan):
            logger.info("Attempting with backup model (gemma)")
            try:
                raw_choice = generate_text(
                    prompt=full_prompt,
                    model='gemma',
                    temperature=CreativityLevel.STRICT.value,
                    max_tokens=512
                )
                llm_choices = self._parse_tool_choice(raw_choice)
                logger.debug(f"Gemma fallback result: {len(llm_choices)} tools")
            except Exception as e:
                logger.warning(f"Gemma fallback failed: {e}")

        # Build lookup for tool matching
        allowed_lookup = {tool.identity: tool for tool in allowed_tools}
        allowed_lookup.update({tool.details.get("id", tool.identity): tool for tool in allowed_tools})

        routed_tools = []
        for choice in llm_choices:
            tool = allowed_lookup.get(choice.tool_id)
            if not tool:
                # Try partial matching
                for tool_id, t in allowed_lookup.items():
                    if choice.tool_id.lower() in tool_id.lower() or tool_id.lower() in choice.tool_id.lower():
                        tool = t
                        break
            
            if not tool:
                logger.warning(f"Unknown tool_id: {choice.tool_id}")
                continue
            
            provided_inputs = choice.inputs if isinstance(choice.inputs, dict) else {}
            context = {"prompt": prompt, "plan": plan}
            
            sig = inspect.signature(tool.__call__)
            prepared_inputs, missing = ParameterMapper.map_parameters(sig, provided_inputs, context)
            
            if missing:
                prepared_inputs.update({k: "" for k in missing})
            
            routed_tools.append({"tool": tool, "inputs": prepared_inputs})

        return routed_tools
    
    def _plan_suggests_tools(self, plan: str) -> bool:
        """Check if the plan text suggests tools should be used."""
        plan_lower = plan.lower()
        tool_keywords = [
            'web search', 'search', 'look up', 'internet', 'online',
            'calculate', 'compute', 'code', 'execute', 'algorithm',
            'creative', 'brainstorm', 'ideas', 'generate',
            'current', 'latest', 'today', 'price', 'weather'
        ]
        return any(kw in plan_lower for kw in tool_keywords)


class WebSearchToolBlock(ToolBlock):
    details: dict = {
        "description": "Retrieves text content from web search results based on a query.",
        "id": "web_search_tool_block",
        "inputs": ["query (str): The search query.", "max_results (int): Maximum number of search results to retrieve."],
        "outputs": ["results (dict): Summarized content from web search results (cited) along with all the links in lists."]
    }

    def __init__(self):
        self.identity = "web_search_tool_block"
    
    def __call__(self, query: str, max_results: int = 5) -> dict:
        try:
            links = retrieve_links(query, max_results=max_results)
            if not links:
                return {"summarize": f"No search results found for query: {query}", "links": [], "error": "No results"}
            
            results = retrieve_text_many(links)
            valid_results = {url: text for url, text in results.items() if text and len(str(text).strip()) > 20}
            
            if not valid_results:
                return {"summarize": f"Search found {len(links)} links but could not extract readable content. URLs: {', '.join(links[:3])}", "links": links, "error": "No readable content"}
            
            content_text = "\n\n".join(
                f"Source URL: {url}\nSnippet: {text[:600]}"
                for url, text in list(valid_results.items())[:5]
            )
            summarize_prompt = (
                f"Summarize the following web search results about '{query}'.\n"
                "Return a concise summary and a list of sources with IDs like S1, S2.\n"
                "Each source must include the URL and a short snippet grounded in the text.\n\n"
                f"{content_text}"
            )
            summary_obj = generate_text(
                prompt=self._with_task_context(summarize_prompt),
                model='gemma',
                schema=WebSearchSummarySchema,
                temperature=CreativityLevel.LOW.value,
                max_tokens=2048
            )
            output = {
                "summary": summary_obj.summary,
                "sources": [s.model_dump() for s in summary_obj.sources],
                "links": links
            }
            return output
        except Exception as e:
            return {"summarize": f"Web search failed: {str(e)}", "links": [], "error": str(e)}


class CodeCreationSchema(BaseModel):
    code: str = Field(description="The textual output from executing the code.")
    packages: list[str] = Field(description="List of required packages to run the code.")
    timeout: int = Field(description="Lenient timeout duration for code execution in seconds.")
    visuals: bool = Field(description="List of visual aids generated (if any) of file .")


class PythonExecutionToolBlock(ToolBlock):
    details: dict = {
        "description": "Executes Python code and returns the output.",
        "id": "python_execution_tool_block",
        "inputs": [
            "goal (str): A detailed description of the code needs to achieve.",
            "visual_aids (bool): Whether to include visual aids such as plots or charts in the output.",],
        "outputs": [
            "output (str): The textual output from executing the code.",
            "visuals (list): List of visual aids generated (if any) of file ."]
    }

    def __init__(self):
        self.identity = "python_execution_tool_block"
    
    def __call__(self, goal: str, visual_aids: bool = False) -> str:
        works = False
        failures = 0
        previous_code = None
        errors = None
        while not works and failures < 3:
            code_prompt = (
                "Generate Python code to achieve the following goal:\n\n" + goal +
                "\n\nSECURITY RESTRICTIONS (code will be rejected if violated):\n"
                "- DO NOT import: os, sys, subprocess, shutil\n"
                "- DO NOT use: exec(), eval(), open(), __import__()\n"
                "- For file operations, use tempfile module only\n" +
                ("\n\nIMPORTANT: Create matplotlib plots/charts as requested. "
                 "DO NOT call plt.show() or plt.savefig() - the system will automatically capture all figures. "
                 "Just create the figure and plot the data. Example:\n"
                 "  import matplotlib.pyplot as plt\n"
                 "  import numpy as np\n"
                 "  x = np.linspace(-10, 10, 100)\n"
                 "  y = x**2\n"
                 "  fig, ax = plt.subplots()\n"
                 "  ax.plot(x, y)\n"
                 "  ax.set_title('f(x) = x^2')\n"
                 "  # DO NOT call plt.show() or plt.savefig()" if visual_aids else "") +
                (f"\n\nPrevious code: {previous_code}" if previous_code else "") +
                (f"\n\nError from previous code: {errors}" if errors else "")
            )
            code_outline = generate_text(
                prompt=self._with_task_context(code_prompt),
                model='nemotron',
                schema=CodeCreationSchema,
                temperature=CreativityLevel.MEDIUM.value,
                max_tokens=2048
            )
            output = python_exec_tool(code_outline.code, code_outline.packages, code_outline.timeout, visual_aids)
            works = output['success']
            previous_code = code_outline.code
            errors = output['error'] if not works else None
            failures += 1
        if failures >= 3:
            return {"output": f"Code execution failed after multiple attempts. DO NOT HIDE THIS FACT. Last error: {errors}", "visuals": [], "plots_base64": []}
        text_output = output['output']
        plots = output['plots']
        plots_base64 = output.get('plots_base64', [])
        logger.info(f"PythonExecutionToolBlock returning: plots={len(plots)}, plots_base64={len(plots_base64)}")
        result = {"output": text_output, "visuals": plots, "plots_base64": plots_base64}
        logger.info(f"PythonExecutionToolBlock result keys: {result.keys()}")
        return result


class CreativeIdeaListSchema(BaseModel):
    ideas: list[str] = Field(description="A list of generated creative ideas with rationale.")


class BestIdeaSelectionSchema(BaseModel):
    best_idea: int = Field(description="The list index of the best idea selected from the list of ideas.")


class CreativeIdeaGeneration(ToolBlock):
    details: dict = {
        "description": "Generates creative ideas based on a given topic.",
        "id": "creative_idea_generation_block",
        "inputs": ["topic (str): The topic to generate ideas for.", "creativity_level (CreativityLevel): The level of creativity to apply."],
        "outputs": ["ideas (str): The generated creative ideas."]
    }

    def __init__(self):
        self.identity = "creative_idea_generation_block"
    
    def __call__(self, topic: str, creativity_level: float = CreativityLevel.HIGH.value) -> str:
        full_prompt = f"Generate 5 creative ideas on the following :\n\n{topic}\n\nProvide a brief rationale for each idea."
        ideas_obj = generate_text(
            prompt=self._with_task_context(full_prompt),
            model='gemma',
            schema=CreativeIdeaListSchema,
            temperature=creativity_level,
            max_tokens=2048
        )
        ideas_list = ideas_obj.ideas
        if not ideas_list:
            return "No ideas generated."
        
        best_idea_index = generate_text(
            prompt=self._with_task_context(
                f"From the following ideas, select the best one and provide its index (0-{len(ideas_list) - 1}):\n\n" +
                "\n".join(f"{i}. {idea}" for i, idea in enumerate(ideas_list))
            ),
            model='nemotron',
            schema=BestIdeaSelectionSchema,
            temperature=CreativityLevel.MEDIUM.value,
        )

        if 0 <= best_idea_index.best_idea < len(ideas_list):
            selected_idea = ideas_list[best_idea_index.best_idea]
        else:
            selected_idea = ideas_list[0]
        return selected_idea


class ResponseSynthesizerBlock(PipelineBlock):
    details: dict = {
        "description": "Synthesizes information from multiple sources into a coherent summary.",
        "id": "response_synthesizer_block",
        "prompt_creation_criteria": "The prompt must instruct the system to combine information from various sources, ensuring coherence and relevance to the original prompt and general adherence plan. The prompt should guide the system to produce a well-structured summary that effectively integrates the gathered information. This must be generalizable and applicable to a wide range of inputs. The prompt may add additional instructions as needed so long as they align with these elements.",
        "inputs": [
            "prompt (str): The base prompt of the user.",
            "tool_responses (dict[str]): A dictionary of information gathered from various tools to synthesize.",
            "plan (str): The plan outlining the synthesis approach."
        ],
        "outputs": ["summary (str): The synthesized and final response to the prompt."]
    }

    def __init__(self):
        self.identity = "response_synthesizer_block"
        self.prompt = get_prompts().get(self.identity, "")
    
    def __call__(self, prompt, sources: dict[str], plan) -> str:
        combined_sources = "\n\n".join(f"Tool used: {i+1}\nContent:\n{content}" for i, content in enumerate(sources.values()) if content)
        prompt = self.prompt.replace("{sources}", combined_sources).replace("{plan}", plan).replace("{prompt}", prompt)
        prompt = self._with_task_context(prompt)
        summary_obj = generate_text(
            prompt=prompt,
            model='gemma',
            schema=SynthesizerSchema,
            temperature=CreativityLevel.LOW.value,
        )
        return {"body": summary_obj.body, "used_sources": summary_obj.used_sources}
