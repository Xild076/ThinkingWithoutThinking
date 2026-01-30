import inspect
import json
from random import random
from src.utility import generate_text, load_prompts, reload_prompts
from src.tool_utility import python_exec_tool, retrieve_text_many, retrieve_links
from src.parameter_mapper import ParameterMapper

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
        plan = generate_text(
            prompt=full_prompt,
            model='gemma',
            temperature=CreativityLevel.MEDIUM.value,
            max_tokens=2048
        )
        return plan


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
        critique = generate_text(
            prompt=full_prompt,
            model='nemotron',
            temperature=CreativityLevel.LOW.value,
            max_tokens=1024
        )
        return critique


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
        improvements = generate_text(
            prompt=full_prompt,
            model='gemma',
            temperature=CreativityLevel.MEDIUM.value,
            max_tokens=2048
        )
        return improvements


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

    def _parse_tool_choice(self, raw: str) -> list[ToolChoice]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            return []

        if isinstance(data, list):
            data = {"tool_choice": data}

        if not isinstance(data, dict) or "tool_choice" not in data:
            return []

        return ToolRouterSchema.model_validate(data).tool_choice
    
    def __call__(self, prompt: str, plan: str, allowed_tools: list[ToolBlock]) -> list[dict]:
        possible_tools = []
        for tool in allowed_tools:
            tool_info = f"Tool Name: {tool.identity}\nDescription: {tool.details['description']}\nRequired Inputs: {', '.join(tool.details['inputs'])}\nRequired Outputs: {', '.join(tool.details['outputs'])}"
            possible_tools.append(tool_info)
        full_prompt = self.prompt.replace("{tools}", "\n\n".join(possible_tools)).replace("{prompt}", prompt).replace("{plan}", plan)
        full_prompt = self._with_task_context(full_prompt)
        try:
            tool_choice = generate_text(
                prompt=full_prompt,
                model='nemotron',
                schema=ToolRouterSchema,
                temperature=CreativityLevel.LOW.value,
            )
            llm_choices = tool_choice.tool_choice
        except Exception:
            raw_choice = generate_text(
                prompt=full_prompt,
                model='nemotron',
                temperature=CreativityLevel.LOW.value,
                max_tokens=512
            )
            llm_choices = self._parse_tool_choice(raw_choice)

        allowed_lookup = {tool.identity: tool for tool in allowed_tools}
        allowed_lookup.update({tool.details.get("id", tool.identity): tool for tool in allowed_tools})

        routed_tools = []
        for choice in llm_choices:
            tool = allowed_lookup.get(choice.tool_id)
            if not tool:
                continue
            provided_inputs = choice.inputs if isinstance(choice.inputs, dict) else {}
            context = {"prompt": prompt, "plan": plan}
            
            sig = inspect.signature(tool.__call__)
            prepared_inputs, missing = ParameterMapper.map_parameters(sig, provided_inputs, context)
            
            if missing:
                prepared_inputs.update({k: "" for k in missing})
            
            routed_tools.append({"tool": tool, "inputs": prepared_inputs})

        return routed_tools


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
            
            content_text = "\n\n".join(f"Source: {url}\n{text[:600]}" for url, text in list(valid_results.items())[:5])
            summarize_prompt = f"Summarize the following web search results about '{query}'. Focus on key facts, cite sources when making specific claims:\n\n{content_text}"
            summarize = generate_text(
                prompt=self._with_task_context(summarize_prompt),
                model='gemma',
                temperature=CreativityLevel.LOW.value,
                max_tokens=2048
            )
            output = {"summarize": summarize, "links": links}
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
                (f"\n\nInclude visual aids (save as temporary files)." if visual_aids else "") +
                (f"\n\n Previous code: {previous_code}" if previous_code else "") +
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
            return {"output": f"Code execution failed after multiple attempts. DO NOT HIDE THIS FACT. Last error: {errors}", "visuals": []}
        text_output = output['output']
        plots = output['plots']
        plots_base64 = output.get('plots_base64', [])
        return {"output": text_output, "visuals": plots, "plots_base64": plots_base64}


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
        summary = generate_text(
            prompt=prompt,
            model='gemma',
            temperature=CreativityLevel.LOW.value,
        )
        return summary
