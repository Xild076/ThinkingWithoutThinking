import sys
import inspect
import json
import re
import time
from urllib.parse import urlparse, urlunparse
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import Literal

try:
    from utility import generate_text, load_prompts, get_prompt, logger
    from tools.web_search_tool import get_links_from_duckduckgo, get_site_details_many
    from tools.python_exec_tool import python_exec_tool
    from tools.wikipedia_tool import get_wikipedia_page_content, search_wikipedia
except Exception:  # pragma: no cover - package import fallback
    from src.utility import generate_text, load_prompts, get_prompt, logger
    from src.tools.web_search_tool import get_links_from_duckduckgo, get_site_details_many
    from src.tools.python_exec_tool import python_exec_tool
    from src.tools.wikipedia_tool import get_wikipedia_page_content, search_wikipedia

from collections.abc import Mapping

load_prompts("data/prompts.json")

WRITER_MODEL = None
ANALYTICAL_MODEL = None

style = "fast"

if style == "fast":
    WRITER_MODEL = "nemotron"
    ANALYTICAL_MODEL = "nemotron"
elif style == "best":
    WRITER_MODEL = "gemma"
    ANALYTICAL_MODEL = "nemotron"
else:
    raise ValueError(f"Unknown style: {style}")

def _get_all_tool_classes():
    tool_classes = []
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_item in classes:
        if issubclass(class_item[1], ToolBlock) and class_item[1] != ToolBlock:
            tool_classes.append(class_item[1])
    return tool_classes

def _get_all_prompted_blocks():
    tool_classes = []
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_item in classes:
        if issubclass(class_item[1], PipelineBlock) and class_item[1] != PipelineBlock and class_item[1] != RouterBlock and class_item[1] != ToolBlock:
            if not issubclass(class_item[1], ToolBlock):
                tool_classes.append(class_item[1])
    return tool_classes
    

def _model_to_primitive(obj: BaseModel) -> dict:
    """Turns BaseModel and all nested BaseModel instances into dicts

    Args:
        obj (BaseModel): Input object to be converted to primitive types

    Returns:
        dict: The input object converted to primitive types (dicts, lists, tuples, sets, and basic types)
    """
    if isinstance(obj, BaseModel):
        return _model_to_primitive(dict(obj))
    if isinstance(obj, Mapping):
        return {k: _model_to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_model_to_primitive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_model_to_primitive(v) for v in obj)
    if isinstance(obj, set):
        return {_model_to_primitive(v) for v in obj}
    return obj

def validate_prompts():
    prompts_with_issues = []
    for block_class in _get_all_prompted_blocks():
        block_details = block_class.details
        try:
            prompt = get_prompt(block_details['id'])
            for input in block_details['inputs']:
                if f"{{{{{input.name}}}}}" not in prompt and f"{{{{{{{input.name}}}}}}}" not in prompt:
                    prompts_with_issues.append((block_details['id'], block_details['name'], f"Missing placeholder for input '{input.name}'"))
        except Exception as e:
            logger.error(f"Error loading prompt for block {block_details['name']} with id {block_details['id']}: {e}")
            prompts_with_issues.append((block_details['id'], block_details['name'], f"Error loading prompt: {e}"))
            continue

    return prompts_with_issues

class CreativityLevel(float, Enum):
    MIN = 0.0
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    MAX = 1.0


class PipelineObject(object):
    def __init__(self, name: str, description: str, type: str):
        self.name = name
        self.description = description
        self.type = type
    
    def __str__(self):
        return f"{self.name}(description={self.description}, type={self.type})"

class PipelineBlock(object):
    details = {
        "id": "base_pipeline_block",
        "name": "Base Pipeline Block",
        "description": "This is the base class for all pipeline blocks.",
        "prompt_creation_parameters": {
            "details": "Details on the creation of a prompt.",
            "objective": "The objective the prompt is trying to achieve.",
            "contains": {"criteria": "The general instructions that must be included in the prompt"},
            "success_rubric": {"criteria": {"description": "Description of criteria for success that must be included in the prompt", "weight": 1.0}},
        },
        "inputs": [PipelineObject("input", "Input data for the block", "any")],
        "outputs": [PipelineObject("output", "Output data from the block", "any")],
        "schema": BaseModel,
        "creativity_level": CreativityLevel,
        "model": Literal["gemma", "nemotron"],
    }

    def __init__(self):
        pass

    def _validate_inputs(self, inputs: dict) -> None:
        expected = {input_obj.name: input_obj.type for input_obj in self.details['inputs']}
        missing = [name for name in expected if name not in inputs]
        extra = [name for name in inputs if name not in expected]

        type_mismatches = []
        for name, expected_type in expected.items():
            if name not in inputs:
                continue
            if expected_type == "any":
                continue
            value = inputs[name]
            expected_py_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
            }.get(expected_type)
            if expected_py_type and not isinstance(value, expected_py_type):
                type_mismatches.append(
                    f"{name} expected {expected_type} got {type(value).__name__}"
                )

        if missing or extra or type_mismatches:
            message = (
                f"Input validation failed. Missing: {missing or 'none'}, "
                f"Extra: {extra or 'none'}, Type mismatches: {type_mismatches or 'none'}"
            )
            logger.error(message)
            raise ValueError(message)

    def _prompt_value(self, value):
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _render_prompt_template(self, prompt_id: str, values: dict[str, object]) -> str:
        prompt = get_prompt(prompt_id)
        for key, value in values.items():
            formatted = self._prompt_value(value)
            prompt = prompt.replace(f"{{{{{{{key}}}}}}}", formatted)
            prompt = prompt.replace(f"{{{{{key}}}}}", formatted)
        return prompt

    def _prompt_builder(self, inputs):
        return self._render_prompt_template(self.details['id'], inputs)

    def process(self, inputs: dict) -> dict:
        block_name = self.details.get('name', 'UnknownBlock')
        logger.info(f"Block process() start: {block_name}")
        
        self._validate_inputs(inputs)
        logger.debug(f"Input validation passed for {block_name}")
        
        prompt = self._prompt_builder(inputs)
        prompt_preview = prompt[:100].replace('\n', ' ')
        logger.debug(f"Prompt built: {prompt_preview}...")
        
        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        output = _model_to_primitive(output)
        output_preview = str(output)[:50]
        logger.info(f"Block process() complete: {block_name} - output: {output_preview}...")

        return output

class ToolBlock(PipelineBlock):
    details = {
        "id": "base_tool_block",
        "name": "Base Tool Block",
        "description": "This is the base class for all tool blocks.",
        "inputs": [PipelineObject("input", "Input data for the tool", "any")],
        "outputs": [PipelineObject("output", "Output data from the tool", "any")],
        "creativity_level": CreativityLevel,
        "schema": BaseModel,
        "model": Literal["gemma", "nemotron"],
    }

    def __init__(self):
        super().__init__()
    
    def process(self, inputs: dict) -> dict:
        raise NotImplementedError("ToolBlock process method must be implemented by subclasses.")

class RouterBlock(PipelineBlock):
    details = {
        "id": "base_router_block",
        "name": "Base Router Block",
        "description": "This is the base class for all router blocks.",
        "prompt_creation_parameters": "Details on the creation of a prompt for routing.",
        "inputs": [PipelineObject("input", "Input data for the router", "any")],
        "outputs": [PipelineObject("routes", "List routes for the router to go to", "list"),
                    PipelineObject("continuity", "Whether to continue routing post routing", "any")],
        "schema": BaseModel,
        "creativity_level": CreativityLevel,
        "model": Literal["gemma", "nemotron"],
    }

    def __init__(self):
        super().__init__()
    
    def _get_available_tools(self):
        tool_classes = _get_all_tool_classes()
        available_tools = []
        for tool_class in tool_classes:
            available_tools.append({
                "id": tool_class.details['id'],
                "name": tool_class.details['name'],
                "description": tool_class.details['description'],
                "inputs": [{"name": input_obj.name, "description": input_obj.description, "type": input_obj.type} for input_obj in tool_class.details['inputs']],
                "outputs": [{"name": output_obj.name, "description": output_obj.description, "type": output_obj.type} for output_obj in tool_class.details['outputs']],
            })
        return available_tools

    def _available_tools_text(self, available_tools):
        available_tools_text = ""
        for tool in available_tools:
            available_tools_text += (
                f"Tool ID: {tool['id']}\n"
                f"Tool Name: {tool['name']}\n"
                f"Description: {tool['description']}\n"
                "Inputs:\n"
            )
            for input_obj in tool['inputs']:
                available_tools_text += f"- {input_obj['name']} ({input_obj['type']}): {input_obj['description']}\n"
            available_tools_text += "Outputs:\n"
            for output_obj in tool['outputs']:
                available_tools_text += f"- {output_obj['name']} ({output_obj['type']}): {output_obj['description']}\n"
            available_tools_text += "\n"
        return available_tools_text
        
    def _parse_routing_output(self, routes):
        parsed_routes = []
        tool_classes = _get_all_tool_classes()
        logger.debug(f"Parsing {len(routes)} routes")
        for i, route in enumerate(routes):
            route_id = route.get("id")
            route_inputs = route.get("inputs")
            if not route_id or not isinstance(route_inputs, dict):
                logger.error(f"Route option is missing 'id' or 'inputs': {route}")
                raise ValueError(f"Route option is missing 'id' or 'inputs': {route}")
            matching_tool_class = next((tool_class for tool_class in tool_classes if tool_class.details['id'] == route_id), None)
            if not matching_tool_class:
                logger.error(f"No matching tool class found for route id: {route_id}")
                raise ValueError(f"No matching tool class found for route id: {route_id}")
            logger.debug(f"Route {i+1}/{len(routes)}: {route_id}")
            parsed_routes.append((matching_tool_class, route_inputs))
        logger.info(f"Parsed {len(parsed_routes)} routes successfully")
        return parsed_routes
    
    def process(self, inputs: dict) -> dict:
        raise NotImplementedError("RouterBlock process method must be implemented by subclasses.")


# General blocks

# Done
class InitialPlanCreationBlockSchema(BaseModel):
    assumed_audience: str = Field(description="The intended audience for the plan")
    assumed_audience_knowledge_level: str = Field(description="The knowledge level of the intended audience")
    assumed_audience_reading_level: str = Field(description="The reading level of the intended audience")
    general_plan: str = Field(description="The generated detailed general plan")
    steps: list[str] = Field(description="The individual steps of the generated plan")
    tool_uses: list[str] = Field(description="Possible tool used in the generated plan")
    complex_response: bool = Field(description="Whether or not each step of the plan is complex enough to warrant subplanning to be comprehensive")
    long_response: bool = Field(description="Whether the final response will long enough to be broken into multiple responses to be aggregated or not")
    response_criteria: list[str] = Field(description="The criteria for a successful response to the plan")

class InitialPlanCreationBlock(PipelineBlock):
    details = {
        "id": "initial_plan_creation_block",
        "name": "Initial Plan Creation Block",
        "description": "Generates a detailed general plan based on the given objective.",
        "prompt_creation_parameters": {
            "details": "Create a detailed general plan to achieve the given objective.",
            "objective": "The objective for which a detailed general plan is to be created.",
            "success_rubric": {
                "accuracy": {
                    "description": "How accurately the plan addresses the objective.",
                    "weight": 0.4
                },
                "feasibility": {
                    "description": "How feasible the plan is to implement.",
                    "weight": 0.3
                },
                "completeness": {
                    "description": "How complete and thorough the plan is.",
                    "weight": 0.3
                }
            }
        },
        "inputs": [PipelineObject("prompt", "The prompt from which a plan to respond is generated", "str"),
                   PipelineObject("available_tools", "The tools available for the plan to use", "list")],
        "outputs": [PipelineObject("plan", "The generated detailed plan", "str"),
                    PipelineObject("steps", "The individual steps of the generated plan", "list"),
                    PipelineObject("tool_uses", "Possible tool used in the generated plan", "list"),
                    PipelineObject("complex_response", "Whether or not each step of the plan is complex enough to warrant subplanning to be comprehensive", "bool"),
                    PipelineObject("long_response", "Whether the final response will long enough to be broken into multiple responses to be aggregated or not", "bool"),
                    PipelineObject("response_criteria", "The criteria for a successful response to the plan", "list")],
        "schema": InitialPlanCreationBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": WRITER_MODEL,
    }

    def __init__(self):
        super().__init__()

# Done
class SubPlanCreationBlockSchema(BaseModel):
    sub_plan: str = Field(default="", description="The generated detailed sub-plan")
    steps: list[str] = Field(
        default_factory=list,
        description="The individual steps of the generated sub-plan to provide necessary information for the success of the main plan",
    )
    tool_uses: list[str] = Field(default_factory=list, description="Possible tool used in the generated sub-plan")

    @model_validator(mode="before")
    @classmethod
    def _normalize_subplan_shapes(cls, payload):
        if not isinstance(payload, dict):
            return payload

        normalized = dict(payload)
        if "sub_plan" not in normalized:
            for alias in ("subplan", "subPlan", "plan"):
                if alias in normalized:
                    normalized["sub_plan"] = normalized.get(alias)
                    break

        def _extract_step_texts(value: object) -> list[str]:
            texts: list[str] = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            texts.append(text)
                    elif isinstance(item, dict):
                        candidate = (
                            item.get("task")
                            or item.get("step")
                            or item.get("description")
                            or item.get("text")
                            or item.get("objective")
                            or ""
                        )
                        text = str(candidate).strip()
                        if text:
                            texts.append(text)
            elif isinstance(value, dict):
                nested = value.get("steps") or value.get("tasks") or value.get("sub_plan")
                texts.extend(_extract_step_texts(nested))
            elif isinstance(value, str):
                text = value.strip()
                if text:
                    texts.append(text)
            return texts

        raw_sub_plan = normalized.get("sub_plan")
        raw_steps = normalized.get("steps")
        sub_plan_steps = _extract_step_texts(raw_sub_plan)
        explicit_steps = _extract_step_texts(raw_steps)

        if explicit_steps:
            normalized["steps"] = explicit_steps
        elif sub_plan_steps:
            normalized["steps"] = sub_plan_steps

        if isinstance(raw_sub_plan, list):
            normalized["sub_plan"] = " ".join(sub_plan_steps).strip()
        elif isinstance(raw_sub_plan, dict):
            candidate = (
                raw_sub_plan.get("summary")
                or raw_sub_plan.get("overview")
                or raw_sub_plan.get("sub_plan")
                or ""
            )
            text = str(candidate).strip()
            if not text and sub_plan_steps:
                text = " ".join(sub_plan_steps).strip()
            normalized["sub_plan"] = text
        elif raw_sub_plan is None:
            normalized["sub_plan"] = ""
        else:
            normalized["sub_plan"] = str(raw_sub_plan).strip()

        tool_uses = normalized.get("tool_uses")
        if isinstance(tool_uses, list):
            normalized["tool_uses"] = [
                str(item).strip()
                for item in tool_uses
                if str(item).strip()
            ]
        elif tool_uses is None:
            normalized["tool_uses"] = []
        else:
            tool_text = str(tool_uses).strip()
            normalized["tool_uses"] = [tool_text] if tool_text else []

        return normalized

    @model_validator(mode="after")
    def _hydrate_missing_subplan_fields(self):
        if not self.sub_plan and self.steps:
            self.sub_plan = " ".join(self.steps).strip()
        if self.sub_plan and not self.steps:
            self.steps = [self.sub_plan]
        return self

class SubPlanCreationBlock(PipelineBlock):
    details = {
        "id": "sub_plan_creation_block",
        "name": "Sub-Plan Creation Block",
        "description": "Generates a detailed sub-plan based on the given portion of a plan and objective.",
        "prompt_creation_parameters": {
            "details": "Create a detailed sub-plan to achieve the given objective based on the provided plan.",
            "objective": "The objective for which a detailed sub-plan is to be created.",
            "success_rubric": {
                "accuracy": {
                    "description": "How accurately the sub-plan addresses the objective and aligns with the main plan.",
                    "weight": 0.5
                },
                "feasibility": {
                    "description": "How feasible the sub-plan is to implement within the context of the main plan.",
                    "weight": 0.3
                },
                "completeness": {
                    "description": "How complete and thorough the sub-plan is in relation to its specific task.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("plan", "The main plan from which a sub-plan is generated", "str"),
               PipelineObject("objective", "The specific step of the plan for which the sub-plan is created", "str"),
               PipelineObject("context", "Structured context from previous subplans and tool outputs to inform continuity", "dict")],
        "outputs": [PipelineObject("sub_plan", "The generated detailed sub-plan", "str"),
                    PipelineObject("steps", "The individual steps of the generated sub-plan to provide necessary information for the success of the main plan", "list"),
                    PipelineObject("tool_uses", "Possible tool used in the generated sub-plan", "list")],
        "schema": SubPlanCreationBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": WRITER_MODEL,
    }

    def __init__(self):
        super().__init__()

# Done
class SelfCritiqueBlockSchema(BaseModel):
    class WeaknessItem(BaseModel):
        type: str = Field(default="", description="Issue severity/type marker such as major or minor")
        description: str = Field(default="", description="Short issue description")
        remediation: str = Field(default="", description="Suggested remediation")

    given_item: str = Field(
        default="",
        description="Summarize the intent and content of the given item to be critiqued",
    )
    general_critique: str = Field(default="", description="The generated critique of the given item")
    list_of_issues: list[str] = Field(
        default_factory=list,
        description="A list of specific issues with the item identified in the critique",
    )
    weaknesses: list[WeaknessItem] = Field(
        default_factory=list,
        description="Compatibility field for alternate critique schema variants",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_issue_shapes(cls, payload):
        if not isinstance(payload, dict):
            return payload

        raw_issues = payload.get("list_of_issues")
        if isinstance(raw_issues, list):
            normalized_issues: list[str] = []
            for issue in raw_issues:
                if isinstance(issue, str):
                    text = issue.strip()
                    if text:
                        normalized_issues.append(text)
                    continue

                if isinstance(issue, dict):
                    issue_type = str(issue.get("type") or issue.get("severity") or "").strip()
                    description = str(
                        issue.get("description")
                        or issue.get("issue")
                        or issue.get("weakness")
                        or ""
                    ).strip()
                    remediation = str(issue.get("remediation") or issue.get("fix") or "").strip()

                    if not description:
                        description = str(issue).strip()

                    if issue_type:
                        description = f"[{issue_type}] {description}"
                    if remediation:
                        description = f"{description} Remediation: {remediation}"
                    if description:
                        normalized_issues.append(description)
                    continue

                text = str(issue).strip()
                if text:
                    normalized_issues.append(text)

            payload["list_of_issues"] = normalized_issues

        return payload

    @model_validator(mode="after")
    def _hydrate_from_weaknesses(self):
        if self.weaknesses and not self.list_of_issues:
            issues: list[str] = []
            for weakness in self.weaknesses:
                if weakness.description:
                    issues.append(weakness.description)
            self.list_of_issues = issues

        if self.weaknesses and not self.general_critique:
            composed = []
            for weakness in self.weaknesses:
                if weakness.description:
                    if weakness.remediation:
                        composed.append(f"{weakness.description} Remediation: {weakness.remediation}")
                    else:
                        composed.append(weakness.description)
            self.general_critique = " ".join(composed).strip()

        if not self.given_item:
            self.given_item = "Critique generated without explicit item summary."

        return self

class SelfCritiqueBlock(PipelineBlock):
    details = {
        "id": "self_critique_block",
        "name": "Self-Critique Block",
        "description": "Generates a critique of the given item based on the provided objective of the item.",
        "prompt_creation_parameters": {
            "details": "Generate a critique of the given item based on the provided objective.",
            "objective": "The objective against which the item is critiqued.",
            "success_rubric": {
                "insightfulness": {
                    "description": "How insightful and constructive the critique is in identifying weaknesses and areas for improvement in the item.",
                    "weight": 0.5
                },
                "relevance": {
                    "description": "How relevant the critique is to the specific item and objective.",
                    "weight": 0.3
                },
                "actionability": {
                    "description": "How actionable the critique is in providing clear guidance for improving the item.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("item", "The item to be critiqued", "str"),
                   PipelineObject("objective", "The objective against which the item is critiqued", "str"),
                   PipelineObject("context", "The context in which the item is being critiqued", "dict")],
        "outputs": [PipelineObject("general_critique", "The generated critique of the item", "str"),
                    PipelineObject("list_of_issues", "A list of specific issues with the item identified in the critique", "list"),
                    PipelineObject("given_item", "A summary of the intent and content of the given item to be critiqued", "str")],
        "schema": SelfCritiqueBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": ANALYTICAL_MODEL,
    }

    def __init__(self):
        super().__init__()

class SynthesisBlockSchema(BaseModel):
    synthesis: str = Field(description="The synthesized output based on the given inputs")

class SynthesisBlock(PipelineBlock):
    details = {
        "id": "synthesis_block",
        "name": "Synthesis Block",
        "description": "Generates a synthesis of the given inputs based on the provided objective and plan.",
        "prompt_creation_parameters": {
            "details": "Generate a synthesis of the given inputs based on the provided objective and plan.",
            "objective": "The objective for which the synthesis is being generated.",
            "success_rubric": {
                "comprehensiveness": {
                    "description": "How comprehensively the synthesis integrates and addresses all relevant aspects of the inputs in relation to the objective.",
                    "weight": 0.5
                },
                "coherence": {
                    "description": "How coherent and logically structured the synthesis is in presenting a unified response to the objective.",
                    "weight": 0.3
                },
                "insightfulness": {
                    "description": "How insightful and original the synthesis is in generating new perspectives or solutions based on the inputs.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("tool_context", "The context obtained from the tool outputs", "dict"),
                   PipelineObject("prompt", "The prompt from which the synthesis is being generated", "dict"),
                   PipelineObject("plan", "The plan for the synthesis", "dict")],
        "outputs": [PipelineObject("synthesis", "The synthesized output based on the given inputs", "str")],
        "schema": SynthesisBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": WRITER_MODEL,
    }

    def __init__(self):
        super().__init__()

# Done
class ImprovementBlock(PipelineBlock):

    details = {
        "id": "improvement_block",
        "name": "Improvement Block",
        "description": "Generates an improved version of the given item based on the provided critique and objective.",
        "prompt_creation_parameters": {
            "details": "Generate an improved version of the given item based on the provided critique and objective.",
            "objective": "The objective for which the item is being improved.",
            "success_rubric": {
                "effectiveness": {
                    "description": "How effectively the improved item addresses the objective and incorporates the critique.",
                    "weight": 0.5
                },
                "feasibility": {
                    "description": "How feasible the improved item is in incorporating the critique and enhancing the original item.",
                    "weight": 0.3
                },
                "coherence": {
                    "description": "How coherent and well-structured the improved item is in comparison to the original item.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("item", "The original item to be improved", "str"),
                   PipelineObject("critique", "The critique based on which the item is improved", "str"),
                   PipelineObject("objective", "The objective for which the item is being improved", "str"),
                   PipelineObject("target_schema", "The schema to use for the improved item (plan or synthesis)", "str")],
        "outputs": [PipelineObject("improved_item", "The improved version of the given item", "dict")],
        "schema": InitialPlanCreationBlockSchema | SynthesisBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": WRITER_MODEL,
    }

    def __init__(self):
        super().__init__()

    def process(self, inputs: dict) -> dict:
        logger.info("ImprovementBlock process() start")
        self._validate_inputs(inputs)

        target_schema = (inputs.get("target_schema") or "").strip().lower()
        schema_lookup = {
            "plan": InitialPlanCreationBlockSchema,
            "synthesis": SynthesisBlockSchema,
        }
        selected_schema = schema_lookup.get(target_schema)
        if not selected_schema:
            logger.error(f"Unknown target_schema '{target_schema}'")
            raise ValueError(
                f"Unknown target_schema '{target_schema}'. Expected 'plan' or 'synthesis'."
            )

        logger.debug(f"ImprovementBlock using schema: {selected_schema.__name__}")
        prompt = self._prompt_builder(inputs)

        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=selected_schema,
            temperature=self.details['creativity_level']
        )

        output = _model_to_primitive(output)
        output_preview = str(output)[:50]
        logger.info(f"ImprovementBlock complete - output: {output_preview}...")

        return {
            "steps_for_improvement": output.get("steps_for_improvement", []),
            "improved_item": output,
        }

# Done
class LongResponseSynthesisBlock(PipelineBlock):
    details = {
        "id": "long_response_synthesis_block",
        "name": "Long Response Synthesis Block",
        "description": "Generates a synthesis of the given inputs based on the provided objective, specifically designed for one portion of long responses that may need to be broken into multiple parts and aggregated.",
        "prompt_creation_parameters": {
            "details": "Generate a synthesis of the given inputs based on the provided objective, specifically designed for one portion of long responses that may need to be broken into multiple parts and aggregated.",
            "objective": "The objective for which the synthesis is being generated.",
            "success_rubric": {
                "Fit": {
                    "description": "How well does the portion fit into the overall synthesis and contribute to achieving the objective?",
                    "weight": 0.5
                },
                "Coherence": {
                    "description": "How coherent and logically structured is the portion in relation to the other portions and the overall synthesis?",
                    "weight": 0.3
                },
                "Insightfulness": {
                    "description": "How insightful and original is the portion in contributing to the overall synthesis and achieving the objective?",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("tool_context", "The context obtained from the tool outputs", "dict"),
                   PipelineObject("prompt", "The prompt from which the synthesis is being generated", "dict"),
                   PipelineObject("plan", "The plan for the synthesis", "dict"),
                   PipelineObject("specific_part_outline", "The outline for the specific part of the synthesis; the specific portion of the overall synthesis that this block is responsible for generating", "str")],
        "outputs": [PipelineObject("synthesis", "The synthesized output of the specific part based on the given inputs", "str")],
        "schema": SynthesisBlockSchema,
        "creativity_level": CreativityLevel.MEDIUM,
        "model": WRITER_MODEL,
    }

# Router blocks

# Done
class RouteOption(BaseModel):
    id: str = Field(description="The id of the route option")
    inputs: dict = Field(
        default_factory=dict,
        description="All necessary inputs for the route option to be executed",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_route_option(cls, payload):
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        if "id" not in normalized:
            for alias in ("tool_id", "tool", "route_id"):
                if alias in normalized:
                    normalized["id"] = normalized.get(alias)
                    break
        if "inputs" not in normalized or not isinstance(normalized.get("inputs"), dict):
            normalized["inputs"] = {}
        return normalized
    
class PrimaryToolRouterBlockSchema(BaseModel):
    routes: list[RouteOption] = Field(
        default_factory=list,
        description="The list of routes for the router to go to",
    )
    continuity: bool = Field(
        default=False,
        description="Whether to continue routing post routing (whether to use the output of the tool to route to other tools)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_router_shapes(cls, payload):
        if not isinstance(payload, dict):
            return payload

        normalized = dict(payload)
        routes = normalized.get("routes")

        # Common drift: model returns a single route object at top-level.
        if routes is None and normalized.get("id"):
            route_inputs = normalized.get("inputs")
            normalized["routes"] = [{
                "id": normalized.get("id"),
                "inputs": route_inputs if isinstance(route_inputs, dict) else {},
            }]
        elif isinstance(routes, dict):
            normalized["routes"] = [routes]
        elif isinstance(routes, str):
            route_id = routes.strip()
            normalized["routes"] = [{"id": route_id, "inputs": {}}] if route_id else []
        elif isinstance(routes, list):
            fixed_routes: list[dict] = []
            for route in routes:
                if isinstance(route, str):
                    route_id = route.strip()
                    if route_id:
                        fixed_routes.append({"id": route_id, "inputs": {}})
                elif isinstance(route, dict):
                    fixed_routes.append(route)
            normalized["routes"] = fixed_routes
        else:
            normalized["routes"] = []

        continuity = normalized.get("continuity", False)
        if isinstance(continuity, str):
            lowered = continuity.strip().lower()
            normalized["continuity"] = lowered in {"1", "true", "yes", "y"}
        else:
            normalized["continuity"] = bool(continuity)

        return normalized

class PrimaryToolRouterBlock(RouterBlock):
    details = {
        "id": "primary_tool_router_block",
        "name": "Primary Tool Router Block",
        "description": "Determines the route to take based on the given input and objective.",
        "prompt_creation_parameters": {
            "details": "Determine the route to take based on the given input and objective.",
            "objective": "The objective for which the route is being determined.",
            "success_rubric": {
                "accuracy": {
                    "description": "How accurately the router determines the appropriate route based on the input and objective.",
                    "weight": 0.5
                },
                "relevance": {
                    "description": "How relevant the determined route is to the specific input and objective.",
                    "weight": 0.3
                },
                "clarity": {
                    "description": "How clear and well-defined the determined route is in terms of the necessary steps and inputs required.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("plan", "The plan based on which the route is determined", "str"),
                   PipelineObject("objective", "The objective for which the route is being determined", "str"),
                   PipelineObject("available_tools", "The tools available for routing decisions", "list")],
        "outputs": [PipelineObject("routes", "The list of routes for the router to go to", "list"),
                    PipelineObject("continuity", "Whether to continue routing post routing (whether to use the output of the tool to route to other tools)", "bool")],
        "schema": PrimaryToolRouterBlockSchema,
        "creativity_level": CreativityLevel.LOW,
        "model": ANALYTICAL_MODEL,
    }

    def process(self, inputs):
        logger.info("PrimaryToolRouterBlock process() start")
        plan = inputs.get("plan")
        objective = inputs.get("objective")
        available_tools = self._available_tools_text(self._get_available_tools())

        prompt = self._prompt_builder({
            "plan": plan,
            "objective": objective,
            "available_tools": available_tools
        })

        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        output = _model_to_primitive(output)
        logger.info(f"PrimaryToolRouterBlock: {len(output.get('routes', []))} routes determined, continuity={output.get('continuity')}")

        return output

# Uses schema from Primary Tool Router since they are functionally the same
class SubToolRouterBlock(RouterBlock):
    details = {
        "id": "secondary_tool_router_block",
        "name": "Secondary Tool Router Block",
        "description": "Determines the route to take based on the given input, objective, and context post tool use.",
        "prompt_creation_parameters": {
            "details": "Determine the route to take based on the given input, objective, and context post tool use.",
            "objective": "The objective for which the route is being determined.",
            "success_rubric": {
                "accuracy": {
                    "description": "How accurately the router determines the appropriate route based on the input, objective, and context.",
                    "weight": 0.5
                },
                "relevance": {
                    "description": "How relevant the determined route is to the specific input, objective, and context.",
                    "weight": 0.3
                },
                "clarity": {
                    "description": "How clear and well-defined the determined route is in terms of the necessary steps and inputs required.",
                    "weight": 0.2
                }
            }
        },
        "inputs": [PipelineObject("tool_output", "The output from the tool based on which the route is determined", "str"),
                   PipelineObject("objective", "The objective for which the route is being determined", "str"),
                   PipelineObject("plan", "The context for which the route is being determined", "dict"),
                   PipelineObject("available_tools", "The tools available for routing decisions", "list")],
        "outputs": [PipelineObject("routes", "The list of routes for the router to go to", "list"),
                    PipelineObject("continuity", "Whether to continue routing post routing (whether to use the output of the tool to route to other tools)", "bool")],
        "schema": PrimaryToolRouterBlockSchema,
        "creativity_level": CreativityLevel.LOW,
        "model": ANALYTICAL_MODEL,
    }

    def process(self, inputs):
        logger.info("SubToolRouterBlock process() start")
        tool_output = inputs.get("tool_output")
        objective = inputs.get("objective")
        plan = inputs.get("plan")
        available_tools = self._available_tools_text(self._get_available_tools())


        prompt = self._prompt_builder({
            "tool_output": tool_output,
            "objective": objective,
            "plan": plan,
            "available_tools": available_tools
        })

        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        output = _model_to_primitive(output)
        logger.info(f"SubToolRouterBlock: {len(output.get('routes', []))} routes determined, continuity={output.get('continuity')}")

        return output


class LargeResponseRouterBlockSchema(BaseModel):
    response_parts: list[str] = Field(
        default_factory=list,
        description="A list of smaller response parts for aggregation with descriptions of each part",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_response_parts(cls, payload):
        if not isinstance(payload, dict):
            return payload
        parts = payload.get("response_parts")
        if not isinstance(parts, list):
            return payload
        normalized: list[str] = []
        for part in parts:
            if isinstance(part, str):
                text = part.strip()
                if text:
                    normalized.append(text)
                continue
            if isinstance(part, dict):
                candidate = (
                    part.get("part")
                    or part.get("title")
                    or part.get("description")
                    or part.get("text")
                    or ""
                )
                text = str(candidate).strip()
                if text:
                    normalized.append(text)
                continue
            text = str(part).strip()
            if text:
                normalized.append(text)
        payload["response_parts"] = normalized
        return payload

class LargeResponseRouterBlock(RouterBlock):
    details = {
        "id": "large_response_router_block",
        "name": "Large Response Router Block",
        "description": "If the response is deemed too long, this block determines how to break the response into smaller parts for aggregation.",
        "inputs": [PipelineObject("plan", "The plan based on which the routing decision is made", "dict"),
                   PipelineObject("objective", "The objective for which the routing decision is being made", "str"),
                   PipelineObject("response_criteria", "The criteria for a successful response to the plan", "list"),
                   PipelineObject("tool_context", "Any context obtained from the tool outputs", "dict")],
        "outputs": [PipelineObject("response_parts", "A list of smaller response parts for aggregation with descriptions of each part", "list")],
        "schema": LargeResponseRouterBlockSchema,
        "creativity_level": CreativityLevel.MIN,
        "model": ANALYTICAL_MODEL,
    }

    def process(self, inputs):
        prompt = self._prompt_builder(inputs)

        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        return dict(output)

# Tool Classes

class WebSearchToolBlockSchema(BaseModel):
    summary: str = Field(description="The relevant information returned from the web search summarized")

class WebSearchToolBlock(ToolBlock):
    details = {
        "id": "web_search_tool_block",
        "name": "Web Search Tool Block",
        "description": "Performs a web search based on the given query and returns relevant information.",
        "prompt_creation_parameters": {
            "details": "Summarize retrieved web evidence grounded strictly in supplied sources.",
            "objective": "Produce concise, evidence-grounded summaries with uncertainty caveats and citation markers.",
            "success_rubric": {
                "evidence_fidelity": {"description": "Uses only provided sources without hallucination.", "weight": 0.45},
                "relevance": {"description": "Prioritizes most relevant evidence to the user query.", "weight": 0.35},
                "clarity": {"description": "Delivers concise, structured synthesis.", "weight": 0.2},
            },
        },
        "inputs": [PipelineObject("query", "The query for which to perform the web search", "str")],
        "outputs": [PipelineObject("cited_summary", "The relevant information returned from the web search summarized", "str"),
                    PipelineObject("search_result_links", "The relevant information returned from the web search in the form of links", "dict")],
        "creativity_level": CreativityLevel.LOW,
        "schema": WebSearchToolBlockSchema,
        "model": ANALYTICAL_MODEL,
    }

    def __init__(self):
        super().__init__()

    def _query_terms(self, query: str) -> list[str]:
        terms = [term for term in re.findall(r"[a-zA-Z0-9]+", (query or "").lower()) if len(term) >= 3]
        return list(dict.fromkeys(terms))

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            normalized = parsed._replace(query="", fragment="")
            return urlunparse(normalized)
        except Exception:
            return url

    def _domain_quality_score(self, url: str) -> int:
        domain = (urlparse(url).netloc or "").lower()
        high_quality_markers = [".edu", "wikipedia.org", "stanford.edu", "britannica.com", "arxiv.org", "springer.com"]
        low_quality_markers = ["pinterest", "reddit", "quora", "fandom", "tiktok", "instagram"]
        score = 0
        if any(marker in domain for marker in high_quality_markers):
            score += 3
        if any(marker in domain for marker in low_quality_markers):
            score -= 2
        return score

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        max_input_chars = 140_000
        max_output_chars = 110_000
        cleaned = " ".join(text[:max_input_chars].split())
        noise_markers = [
            "cookie",
            "privacy policy",
            "terms of use",
            "subscribe",
            "sign in",
            "jump to content",
            "main menu",
        ]
        for marker in noise_markers:
            cleaned = re.sub(rf"\b{re.escape(marker)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split()).strip()
        if len(cleaned) > max_output_chars:
            cleaned = cleaned[:max_output_chars]
        return cleaned

    def _score_relevance(self, query_terms: list[str], title: str, text: str) -> int:
        haystack = f"{title or ''} {text or ''}".lower()
        if not haystack.strip():
            return 0
        score = 0
        for term in query_terms:
            if term in haystack:
                score += 1
        if query_terms and all(term in haystack for term in query_terms[: min(2, len(query_terms))]):
            score += 2
        return score

    def _relevant_excerpt(self, text: str, query_terms: list[str], max_chars: int) -> str:
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected: list[str] = []
        total_len = 0

        for idx, sentence in enumerate(sentences):
            lowered = sentence.lower()
            matches = not query_terms or any(term in lowered for term in query_terms)
            if not matches:
                continue

            for neighbor_idx in (idx - 1, idx, idx + 1):
                if neighbor_idx < 0 or neighbor_idx >= len(sentences):
                    continue
                neighbor = sentences[neighbor_idx].strip()
                if not neighbor:
                    continue
                if neighbor in selected:
                    continue
                if total_len + len(neighbor) + 1 > max_chars:
                    break
                selected.append(neighbor)
                total_len += len(neighbor) + 1

            if total_len >= max_chars:
                break

        if not selected:
            return text[:max_chars]
        return " ".join(selected)
    
    def process(self, inputs):
        query = inputs.get("query")
        logger.info(f"WebSearchToolBlock: searching for '{query}'")
        process_started = time.perf_counter()

        max_sources = 5
        max_chars_per_source = 1800
        max_total_prompt_chars = 12000
        min_relevance_score = 1
        max_urls_to_fetch = 12
        max_per_domain = 2
        preprocess_budget_seconds = 12.0

        query_terms = self._query_terms(query)

        retrieval_started = time.perf_counter()
        urls = get_links_from_duckduckgo(query)
        logger.debug(f"WebSearchToolBlock: found {len(urls)} URLs")
        outputs = get_site_details_many(urls[:max_urls_to_fetch])
        retrieval_elapsed = time.perf_counter() - retrieval_started
        logger.debug(
            f"WebSearchToolBlock: retrieved {len(outputs)} site details in {retrieval_elapsed:.2f}s"
        )

        ranked_outputs = []
        seen_urls = set()
        domain_counts: dict[str, int] = {}
        preprocess_started = time.perf_counter()
        for output in outputs:
            if time.perf_counter() - preprocess_started > preprocess_budget_seconds:
                logger.warning(
                    "WebSearchToolBlock: preprocessing budget exceeded; continuing with partial ranked sources"
                )
                break
            raw_url = str(output.get("url") or "").strip()
            url = self._normalize_url(raw_url)
            title = str(output.get("title") or "").strip()
            raw_text = str(output.get("text") or "")
            text = self._clean_text(raw_text)
            if not url or url in seen_urls or not text:
                continue

            domain = (urlparse(url).netloc or "").lower()
            current_domain_count = domain_counts.get(domain, 0)
            if current_domain_count >= max_per_domain:
                continue

            seen_urls.add(url)
            domain_counts[domain] = current_domain_count + 1

            relevance_score = self._score_relevance(query_terms, title, text)
            if relevance_score < min_relevance_score:
                continue

            quality_score = self._domain_quality_score(url)
            total_score = relevance_score * 2 + quality_score

            ranked_outputs.append(
                {
                    "url": url,
                    "title": title,
                    "text": text,
                    "score": total_score,
                }
            )

        preprocess_elapsed = time.perf_counter() - preprocess_started
        logger.debug(
            f"WebSearchToolBlock: preprocessing complete in {preprocess_elapsed:.2f}s, candidates={len(ranked_outputs)}"
        )
        ranked_outputs.sort(key=lambda item: item["score"], reverse=True)
        ranked_outputs = ranked_outputs[:max_sources]

        if not ranked_outputs:
            fallback_outputs = []
            seen_fallback = set()
            for output in outputs:
                url = self._normalize_url(str(output.get("url") or "").strip())
                if not url or url in seen_fallback:
                    continue
                title = str(output.get("title") or "").strip()
                text = self._clean_text(str(output.get("text") or ""))
                if not text:
                    continue
                seen_fallback.add(url)
                fallback_outputs.append({"url": url, "title": title, "text": text, "score": 0})
                if len(fallback_outputs) >= 2:
                    break
            ranked_outputs = fallback_outputs
        
        prompt = ""
        url_numbered = {}

        for i, output in enumerate(ranked_outputs):
            source_text = self._relevant_excerpt(output.get("text", ""), query_terms, max_chars_per_source)
            source_block = (
                f"Title: {output.get('title')}\n"
                f"Text: {source_text}\n"
                f"URL: {output.get('url')}\n"
                f"Reference ID: {i+1}\n"
            )

            if len(prompt) + len(source_block) > max_total_prompt_chars:
                logger.warning("WebSearchToolBlock: reached prompt size cap while assembling sources")
                break

            prompt += source_block
            url_numbered[i+1] = output.get("url")

        if not prompt.strip():
            logger.warning("WebSearchToolBlock: no sufficiently relevant sources found after filtering")
            return {
                "cited_summary": "No relevant information found.",
                "search_result_links": {},
            }
        
        logger.debug(
            f"WebSearchToolBlock: prompt assembled with {len(url_numbered)} sources, chars={len(prompt)}"
        )
        output_contract = json.dumps(
            {
                "summary": "string",
            },
            indent=2,
        )
        prompt += "\n\n" + self._render_prompt_template(
            self.details["id"],
            {
                "query": query,
                "output_contract": output_contract,
            },
        )

        output = generate_text(
            prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        results = {
            "cited_summary": output.summary,
            "search_result_links": url_numbered
        }
        summary_preview = output.summary[:50]
        total_elapsed = time.perf_counter() - process_started
        logger.info(
            f"WebSearchToolBlock: search complete in {total_elapsed:.2f}s - summary: {summary_preview}..."
        )

        return results


class PythonCodeExecutionToolBlockSchema(BaseModel):
    code_to_run: str = Field(description="The Python code to be executed")
    packages_needed: list[str] = Field(description="The packages needed to run the code")

class PythonCodeExecutionToolBlock(ToolBlock):
    details = {
        "id": "python_code_execution_tool_block",
        "name": "Python Code Execution Tool Block",
        "description": "Executes the given Python code and returns the output.",
        "prompt_creation_parameters": {
            "details": "Generate deterministic Python code with a strict JSON output contract and repair loop.",
            "objective": "Solve objective-specific computation tasks safely while minimizing retries and runtime.",
            "success_rubric": {
                "schema_fidelity": {"description": "Returns valid JSON fields code_to_run/packages_needed.", "weight": 0.35},
                "execution_reliability": {"description": "Code executes successfully with bounded complexity.", "weight": 0.4},
                "safety_and_constraints": {"description": "Respects visual and import constraints.", "weight": 0.25},
            },
        },
        "inputs": [PipelineObject("objective", "The objective of the code execution", "str"),
                   PipelineObject("visuals_needed", "Whether the code needs to create visuals or not", "bool")],
        "outputs": [PipelineObject("results", "The output of the code execution", "str"),
                    PipelineObject("plots", "Any errors that occurred during code execution", "str")],
        "creativity_level": CreativityLevel.MIN,
        "schema": PythonCodeExecutionToolBlockSchema,
        "model": ANALYTICAL_MODEL,
    }

    def __init__(self):
        super().__init__()

    def _infer_package_hints(self, objective: str, visuals_needed: bool) -> list[str]:
        objective_lower = (objective or "").lower()
        hints = []

        if any(token in objective_lower for token in ["symbolic", "derive", "algebra", "trigonometric", "coefficient", "proof"]):
            hints.append("sympy")
        if any(token in objective_lower for token in ["array", "matrix", "linear algebra", "numerical"]):
            hints.append("numpy")
        if any(token in objective_lower for token in ["table", "csv", "dataframe", "dataset"]):
            hints.append("pandas")
        if any(token in objective_lower for token in ["regression", "distribution", "hypothesis", "statistical", "stats"]):
            hints.extend(["scipy", "statsmodels"])
        if any(token in objective_lower for token in ["crawl", "scrape", "html", "web page"]):
            hints.extend(["requests", "beautifulsoup4"])

        if visuals_needed:
            if "interactive" in objective_lower:
                hints.append("plotly")
            else:
                hints.append("matplotlib")

        deduped = []
        seen = set()
        for package in hints:
            if package not in seen:
                seen.add(package)
                deduped.append(package)
        return deduped

    def _sanitize_package_plan(self, suggested_packages: list[str], objective: str, code: str, visuals_needed: bool) -> list[str]:
        objective_hints = set(self._infer_package_hints(objective, visuals_needed))
        normalized = []
        for package in suggested_packages or []:
            if isinstance(package, str) and package.strip():
                normalized.append(package.strip().lower())

        code_imports = set()
        for match in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", code or "", flags=re.MULTILINE):
            code_imports.add(match.group(1).split(".")[0].lower())

        import_map = {
            "bs4": "beautifulsoup4",
            "sklearn": "scikit-learn",
        }

        combined = set(normalized) | objective_hints | {import_map.get(item, item) for item in code_imports}
        if not visuals_needed:
            combined -= {"matplotlib", "seaborn", "plotly"}
        allowlist = {
            "numpy", "pandas", "matplotlib", "seaborn", "plotly", "scipy", "sympy",
            "scikit-learn", "requests", "beautifulsoup4", "pillow", "statsmodels",
        }
        return sorted([package for package in combined if package in allowlist])

    def _contains_disallowed_visual_code(self, code: str) -> bool:
        patterns = [
            r"^\s*import\s+matplotlib\b",
            r"^\s*from\s+matplotlib\b",
            r"^\s*import\s+seaborn\b",
            r"^\s*from\s+seaborn\b",
            r"^\s*import\s+plotly\b",
            r"^\s*from\s+plotly\b",
            r"\bplt\.show\s*\(",
            r"\bplt\.savefig\s*\(",
            r"\bpx\.",
        ]
        return any(re.search(pattern, code or "", flags=re.MULTILINE) for pattern in patterns)

    def _execution_timeout(self, objective: str, visuals_needed: bool) -> int:
        objective_lower = (objective or "").lower()
        if any(token in objective_lower for token in ["expand", "coefficient", "symbolic", "derive"]):
            return 25
        if visuals_needed:
            return 30
        return 20
    
    def process(self, inputs, retries=3):
        objective = inputs.get("objective") or ""
        visuals_needed = bool(inputs.get("visuals_needed"))
        logger.info(f"PythonCodeExecutionToolBlock: objective='{objective[:50]}...', visuals={visuals_needed}")

        package_hints = self._infer_package_hints(objective, visuals_needed)
        package_hint_text = ", ".join(package_hints) if package_hints else "none"
        
        errors = None
        prev_code = None
        execution_result = {
            "success": False,
            "output": "",
            "plots": [],
            "error": "Code generation/execution did not run.",
        }
        attempt_log = []
        retry = 0
        while retry < retries: 
            logger.debug(f"PythonCodeExecutionToolBlock: attempt {retry+1}/{retries}")
            output_contract = json.dumps(
                {
                    "code_to_run": "string",
                    "packages_needed": ["string"],
                },
                indent=2,
            )
            prompt = self._render_prompt_template(
                self.details["id"],
                {
                    "objective": objective,
                    "visuals_needed": visuals_needed,
                    "package_hints": package_hint_text,
                    "previous_error": errors or "none",
                    "previous_code": prev_code or "",
                    "output_contract": output_contract,
                },
            )

            generated_plan = generate_text(
                prompt=prompt,
                model=self.details['model'],
                schema=self.details['schema'],
                temperature=self.details['creativity_level'],
                retries=3,
                retry_delay=0.8,
                max_total_retry_wait=12.0,
            )
            logger.debug(f"Code generated: {generated_plan.code_to_run[:50]}...")

            if not visuals_needed and self._contains_disallowed_visual_code(generated_plan.code_to_run):
                errors = "Generated code used visualization libraries/functions while visuals_needed=False."
                prev_code = generated_plan.code_to_run
                attempt_log.append(
                    {
                        "attempt": retry + 1,
                        "selected_packages": [],
                        "generated_code": generated_plan.code_to_run,
                        "success": False,
                        "error": errors,
                        "output_preview": "",
                        "plots": [],
                    }
                )
                logger.warning(f"PythonCodeExecutionToolBlock: attempt {retry+1} rejected - visuals disabled but plotting code present")
                retry += 1
                continue

            selected_packages = self._sanitize_package_plan(
                suggested_packages=generated_plan.packages_needed,
                objective=objective,
                code=generated_plan.code_to_run,
                visuals_needed=visuals_needed,
            )

            attempt_timeout = self._execution_timeout(objective, visuals_needed)
            exec_start = time.perf_counter()
            logger.debug(
                "PythonCodeExecutionToolBlock: executing attempt %s with timeout=%ss packages=%s",
                retry + 1,
                attempt_timeout,
                selected_packages,
            )
            execution_result = python_exec_tool(
                generated_plan.code_to_run,
                install_packages=selected_packages,
                timeout=attempt_timeout,
                save_plots=visuals_needed,
            )
            exec_elapsed = time.perf_counter() - exec_start
            logger.info(
                "PythonCodeExecutionToolBlock: execution attempt %s finished in %.2fs (success=%s)",
                retry + 1,
                exec_elapsed,
                bool(execution_result.get("success")),
            )
            attempt_log.append(
                {
                    "attempt": retry + 1,
                    "selected_packages": selected_packages,
                    "generated_code": generated_plan.code_to_run,
                    "success": bool(execution_result.get("success")),
                    "error": execution_result.get("error"),
                    "output_preview": str(execution_result.get("output", ""))[:800],
                    "plots": execution_result.get("plots", []),
                }
            )

            if execution_result.get('success'):
                logger.info(f"PythonCodeExecutionToolBlock: execution successful")
                errors = None
                break

            errors = execution_result.get('error')
            prev_code = generated_plan.code_to_run
            logger.warning(f"PythonCodeExecutionToolBlock: attempt {retry+1} failed - {str(errors)[:50]}...")
            retry += 1

        results = execution_result.get('output', '')
        if errors:
            if results:
                results += "\n"
            results += "The code execution failed after multiple attempts. Ensure that the user is aware of the failures and potential resulting inaccuracies."
        plots = execution_result.get('plots', [])

        return {
            "results": results,
            "plots": plots,
            "diagnostics": {
                "objective": objective,
                "visuals_needed": visuals_needed,
                "package_hints": package_hints,
                "attempts_allowed": retries,
                "attempts_run": len(attempt_log),
                "final_success": bool(execution_result.get("success")),
                "final_error": errors or execution_result.get("error"),
                "attempt_log": attempt_log,
            },
        }
        

class WikipediaSearchToolBlockSchema(BaseModel):
    summary: str = Field(description="The relevant information returned from the Wikipedia search summarized")

class WikipediaSearchToolBlock(ToolBlock):
    details = {
        "id": "wikipedia_search_tool_block",
        "name": "Wikipedia Search Tool Block",
        "description": "Performs a search on Wikipedia based on the given query and returns relevant information.",
        "prompt_creation_parameters": {
            "details": "Summarize Wikipedia content with explicit relevance checks and fallback wording.",
            "objective": "Provide concise, query-aligned summaries and state insufficient relevance explicitly.",
            "success_rubric": {
                "query_alignment": {"description": "Summary stays directly tied to query intent.", "weight": 0.45},
                "faithfulness": {"description": "No fabricated claims beyond provided Wikipedia content.", "weight": 0.35},
                "fallback_quality": {"description": "Signals insufficient relevance clearly when needed.", "weight": 0.2},
            },
        },
        "inputs": [PipelineObject("query", "The query for which to perform the Wikipedia search", "str")],
        "outputs": [PipelineObject("summary", "The relevant information returned from the Wikipedia search summarized", "str"),
                    PipelineObject("url", "The URL of the Wikipedia page from which the information was retrieved", "str")],
        "creativity_level": CreativityLevel.LOW,
        "schema": WikipediaSearchToolBlockSchema,
        "model": ANALYTICAL_MODEL,
    }

    def __init__(self):
        super().__init__()
    
    def process(self, inputs, fast_mode=False):
        query = inputs.get("query")
        logger.info(f"WikipediaSearchToolBlock: searching for '{query}', fast_mode={fast_mode}")

        page = search_wikipedia(query)
        logger.debug(f"WikipediaSearchToolBlock: found page '{page}'")

        content = get_wikipedia_page_content(page)
        output_contract = json.dumps({"summary": "string"}, indent=2)

        if fast_mode:
            logger.debug(f"WikipediaSearchToolBlock: using fast mode summarization")
            prompt = self._render_prompt_template(
                self.details["id"],
                {
                    "query": query,
                    "content": content.get("content", ""),
                    "mode": "fast",
                    "output_contract": output_contract,
                },
            )

            output = generate_text(
                prompt=prompt,
                model=self.details['model'],
                schema=self.details['schema'],
                temperature=self.details['creativity_level']
            )

            summary = output.summary
        else:
            summary = content.get("summary")

            query_terms = {term for term in re.findall(r"[a-zA-Z0-9]+", (query or "").lower()) if len(term) >= 4}
            summary_lower = (summary or "").lower()
            if query_terms and not any(term in summary_lower for term in query_terms):
                prompt = self._render_prompt_template(
                    self.details["id"],
                    {
                        "query": query,
                        "content": content.get("content", ""),
                        "mode": "relevance_repair",
                        "output_contract": output_contract,
                    },
                )
                output = generate_text(
                    prompt=prompt,
                    model=self.details['model'],
                    schema=self.details['schema'],
                    temperature=self.details['creativity_level']
                )
                summary = output.summary

        summary_preview = summary[:50]
        logger.info(f"WikipediaSearchToolBlock: complete - summary: {summary_preview}...")

        return {
            "summary": summary,
            "url": content.get("url")
        }


class CreativeIdeaGeneratorSchemaToolBlock(BaseModel):
    ideas: list[str] = Field(description="A list of creative ideas generated based on the objective")

class CreativeIdeaGeneratorToolBlock(ToolBlock):
    details = {
        "id": "creative_idea_generator_tool_block",
        "name": "Creative Idea Generator Tool Block",
        "description": "Generates creative ideas based on the given, art-related objective.",
        "prompt_creation_parameters": {
            "details": "Generate diverse idea sets with explicit feasibility tagging.",
            "objective": "Produce practical and novel ideas without duplication.",
            "success_rubric": {
                "diversity": {"description": "Ideas span distinct creative directions.", "weight": 0.4},
                "feasibility": {"description": "Each idea includes concrete feasibility signal.", "weight": 0.3},
                "novelty": {"description": "Includes high-variance, non-generic options.", "weight": 0.3},
            },
        },
        "inputs": [PipelineObject("objective", "The objective for which to generate creative ideas", "str")],
        "outputs": [PipelineObject("ideas", "A list of creative ideas generated based on the objective", "list")],
        "creativity_level": CreativityLevel.HIGH,
        "schema": CreativeIdeaGeneratorSchemaToolBlock,
        "model": WRITER_MODEL,
    }

    def __init__(self):
        super().__init__()
    
    def process(self, inputs):
        objective = inputs.get("objective") or ""
        logger.info(f"CreativeIdeaGeneratorToolBlock: generating ideas for '{objective[:50]}...'")

        prompt = self._render_prompt_template(
            self.details["id"],
            {
                "objective": objective,
                "output_contract": json.dumps({"ideas": ["string"]}, indent=2),
            },
        )

        output = generate_text(
            prompt=prompt,
            model=self.details['model'],
            schema=self.details['schema'],
            temperature=self.details['creativity_level']
        )

        logger.info(f"CreativeIdeaGeneratorToolBlock: generated {len(output.ideas)} ideas")

        return {
            "ideas": output.ideas
        }


class DeductiveReasoningPremiseToolBlockSchema(BaseModel):
    reasoning: str = Field(description="The reasoning process used to generate the premises")
    premises: list[str] = Field(description="A list of premises used in the deductive reasoning process")

class DeductiveReasoningConfirmPremiseToolBlockSchema(BaseModel):
    reasoning: str = Field(description="The reasoning process used to confirm or deny the validity of each premise")
    premise_validations: list[bool] = Field(description="A list of booleans indicating whether each premise is valid or not")

class DeductiveReasoningConclusionToolBlockSchema(BaseModel):
    reasoning: str = Field(description="The reasoning process used to arrive at the conclusion based on the premises")
    conclusion: str = Field(description="The conclusion arrived at based on the premises")

class DeductiveReasoningConclusionConfirmationToolBlockSchema(BaseModel):
    reasoning: str = Field(description="The reasoning process used to confirm or deny the validity of the conclusion")
    conclusion_valid: bool = Field(description="A boolean indicating whether the conclusion is valid or not")

class DeductiveReasoningToolBlock(ToolBlock):
    details = {
        "id": "deductive_reasoning_premise_tool_block",
        "name": "Deductive Reasoning Premise Tool Block",
        "description": "Generates premises based on the given objective for use in a deductive reasoning process.",
        "prompt_creation_parameters": {
            "details": "Generate and validate premises, then derive conclusion constrained to validated premises.",
            "objective": "Increase reasoning trace reliability with explicit premise and conclusion checks.",
            "success_rubric": {
                "premise_validity": {"description": "Premises are precise, non-redundant, and checkable.", "weight": 0.35},
                "logical_consistency": {"description": "Conclusion depends only on validated premises.", "weight": 0.45},
                "explanatory_clarity": {"description": "Reasoning text is concise and easy to audit.", "weight": 0.2},
            },
        },
        "inputs": [PipelineObject("objective", "The objective of which deductive reasoning is needed to solve.", "str")],
        "outputs": [PipelineObject("premise_reasoning", "The reasoning process used to generate the premises", "str"),
                    PipelineObject("premises", "A list of premises used in the deductive reasoning process", "list"),
                    PipelineObject("conclusion_reasoning", "The reasoning process used to arrive at the conclusion based on the premises", "str"),
                    PipelineObject("conclusion", "The conclusion arrived at based on the premises", "str")],
        "creativity_level": CreativityLevel.MEDIUM,
        "schema": [DeductiveReasoningPremiseToolBlockSchema,
                   DeductiveReasoningConclusionToolBlockSchema,
                   DeductiveReasoningConclusionConfirmationToolBlockSchema,
                   DeductiveReasoningConfirmPremiseToolBlockSchema],
        "model": ANALYTICAL_MODEL,
    }

    def __init__(self):
        super().__init__()
    
    def process(self, inputs):
        objective = inputs.get("objective") or ""
        logger.info(f"DeductiveReasoningPremiseToolBlock: generating premises for '{objective[:50]}...'")

        prompt_premise = self._render_prompt_template(
            "deductive_reasoning_premise_tool_block",
            {
                "objective": objective,
                "output_contract": json.dumps(
                    {
                        "reasoning": "string",
                        "premises": ["string"],
                    },
                    indent=2,
                ),
            },
        )

        output = generate_text(
            prompt=prompt_premise,
            model=self.details['model'],
            schema=self.details['schema'][0],  # Use the DeductiveReasoningPremiseToolBlockSchema for this step
            temperature=self.details['creativity_level']
        )

        logger.info(f"DeductiveReasoningPremiseToolBlock: generated {len(output.premises)} premises")

        prompt_confirm_premise = self._render_prompt_template(
            "deductive_reasoning_confirm_premise_tool_block",
            {
                "objective": objective,
                "premises": output.premises,
                "validated_premises_instruction": "Only validate against objective-grounded criteria.",
                "output_contract": json.dumps(
                    {
                        "reasoning": "string",
                        "premise_validations": ["boolean"],
                    },
                    indent=2,
                ),
            },
        )

        output_confirm = generate_text(
            prompt=prompt_confirm_premise,
            model=self.details['model'],
            schema=self.details['schema'][3],  # Use the DeductiveReasoningConfirmPremiseToolBlockSchema for this step
            temperature=self.details['creativity_level']
        )

        premise_validations = list(output_confirm.premise_validations or [])
        if len(premise_validations) < len(output.premises):
            premise_validations.extend([False] * (len(output.premises) - len(premise_validations)))
        premise_validations = premise_validations[:len(output.premises)]

        premise_confirmed = []
        for premise_text, is_valid in zip(output.premises, premise_validations):
            logger.info(f"Premise: '{premise_text[:50]}...', Valid: {is_valid}")
            premise_confirmed.append({"premise": premise_text, "valid": is_valid})

        valid_premises = [item["premise"] for item in premise_confirmed if item["valid"]]
        if not valid_premises:
            valid_premises = output.premises[:1]

        prompt_conclusion = self._render_prompt_template(
            "deductive_reasoning_conclusion_tool_block",
            {
                "objective": objective,
                "valid_premises": valid_premises,
                "validated_premises_instruction": "Use only validated premises; do not introduce new assumptions.",
                "output_contract": json.dumps(
                    {
                        "reasoning": "string",
                        "conclusion": "string",
                    },
                    indent=2,
                ),
            },
        )

        output_conclusion = generate_text(
            prompt=prompt_conclusion,
            model=self.details['model'],
            schema=self.details['schema'][1],  # Use the DeductiveReasoningConclusionToolBlockSchema for this step
            temperature=self.details['creativity_level']
        )

        prompt_confirm_conclusion = self._render_prompt_template(
            "deductive_reasoning_conclusion_confirmation_tool_block",
            {
                "objective": objective,
                "conclusion": output_conclusion.conclusion,
                "validated_premises_instruction": "Check logical dependence on validated premises only.",
                "output_contract": json.dumps(
                    {
                        "reasoning": "string",
                        "conclusion_valid": "boolean",
                    },
                    indent=2,
                ),
            },
        )

        output_confirm_conclusion = generate_text(
            prompt=prompt_confirm_conclusion,
            model=self.details['model'],
            schema=self.details['schema'][2],  # Use the DeductiveReasoningConclusionConfirmationToolBlockSchema for this step
            temperature=self.details['creativity_level']
        )

        return {
            "premise_reasoning": output.reasoning,
            "premises": output.premises,
            "premise_confirmed": premise_confirmed,
            "conclusion_reasoning": output_conclusion.reasoning,
            "conclusion": output_conclusion.conclusion,
            "conclusion_valid": output_confirm_conclusion.conclusion_valid
        }


def get_tool_prompt_descriptors() -> dict[str, dict[str, object]]:
    return {
        "web_search_tool_block": {
            "id": "web_search_tool_block",
            "name": "Web Search Tool Prompt",
            "schema": WebSearchToolBlockSchema,
            "prompt_creation_parameters": WebSearchToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "wikipedia_search_tool_block": {
            "id": "wikipedia_search_tool_block",
            "name": "Wikipedia Search Tool Prompt",
            "schema": WikipediaSearchToolBlockSchema,
            "prompt_creation_parameters": WikipediaSearchToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "python_code_execution_tool_block": {
            "id": "python_code_execution_tool_block",
            "name": "Python Code Execution Tool Prompt",
            "schema": PythonCodeExecutionToolBlockSchema,
            "prompt_creation_parameters": PythonCodeExecutionToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "creative_idea_generator_tool_block": {
            "id": "creative_idea_generator_tool_block",
            "name": "Creative Idea Generator Tool Prompt",
            "schema": CreativeIdeaGeneratorSchemaToolBlock,
            "prompt_creation_parameters": CreativeIdeaGeneratorToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "deductive_reasoning_premise_tool_block": {
            "id": "deductive_reasoning_premise_tool_block",
            "name": "Deductive Premise Tool Prompt",
            "schema": DeductiveReasoningPremiseToolBlockSchema,
            "prompt_creation_parameters": DeductiveReasoningToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "deductive_reasoning_confirm_premise_tool_block": {
            "id": "deductive_reasoning_confirm_premise_tool_block",
            "name": "Deductive Premise Validation Prompt",
            "schema": DeductiveReasoningConfirmPremiseToolBlockSchema,
            "prompt_creation_parameters": DeductiveReasoningToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "deductive_reasoning_conclusion_tool_block": {
            "id": "deductive_reasoning_conclusion_tool_block",
            "name": "Deductive Conclusion Prompt",
            "schema": DeductiveReasoningConclusionToolBlockSchema,
            "prompt_creation_parameters": DeductiveReasoningToolBlock.details.get("prompt_creation_parameters", {}),
        },
        "deductive_reasoning_conclusion_confirmation_tool_block": {
            "id": "deductive_reasoning_conclusion_confirmation_tool_block",
            "name": "Deductive Conclusion Validation Prompt",
            "schema": DeductiveReasoningConclusionConfirmationToolBlockSchema,
            "prompt_creation_parameters": DeductiveReasoningToolBlock.details.get("prompt_creation_parameters", {}),
        },
    }


validate_prompts()
