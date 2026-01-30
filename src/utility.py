from typing import Any, Literal, Type, TypeVar
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
import google.genai as genai
from openai import OpenAI
import dotenv
import os
import json
import time
import random

from src.cost_tracker import cost_tracker

Schema = genai.types.Schema
T = TypeVar("T", bound=BaseModel)

TEMPERATURE_LIMITS: dict[str, tuple[float, float]] = {
    "gemma": (0.0, 2.0),
    "nemotron": (0.0, 1.0),
}

dotenv.load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY environment variable not set")


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "rate limit" in message or "429" in message or "too many requests" in message


def _is_transient_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ["503", "unavailable", "overloaded", "timeout", "temporarily", "connection"]
    )


def _default_for_type(annotation: Any) -> Any:
    """Get default value for a type annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return []
    if origin is dict:
        return {}
    if annotation in (str,):
        return ""
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if annotation in (bool,):
        return False
    return None


def _schema_fallback(schema: Type[T]) -> T:
    """Create a fallback instance with default values when parsing fails."""
    values: dict[str, Any] = {}
    for name, field_info in schema.model_fields.items():
        if field_info.default is not PydanticUndefined:
            values[name] = field_info.default
        else:
            values[name] = _default_for_type(field_info.annotation)
    return schema.model_validate(values)


def _clamp_temperature(model: str, temperature: float) -> float:
    """Clamp temperature to valid range for the given model."""
    limits = TEMPERATURE_LIMITS.get(model, (0.0, 1.0))
    return max(limits[0], min(limits[1], temperature))


def _estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def generate_text(
    prompt: str,
    model: Literal["gemma", "nemotron"] = "gemma",
    schema: Type[T] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_retries: int = 4,
    max_prompt_chars: int = 12000
) -> str | T:
    """Generate text using Google GenAI or NVIDIA models.
    
    Args:
        prompt: Input text prompt
        model: Model identifier ('gemma' or 'nemotron')
        schema: Optional Pydantic model for response validation
        temperature: Sampling temperature (auto-clamped per model)
        max_tokens: Maximum output tokens
        max_retries: Number of retry attempts for transient errors
        max_prompt_chars: Maximum prompt length before truncation

    Returns:
        Generated text string or validated Pydantic model instance
    """
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]

    temperature = _clamp_temperature(model, temperature)
    
    text: str | None = None
    input_tokens = _estimate_tokens(prompt)
    start_time = time.time()
    error_msg = ""

    for attempt in range(max_retries):
        try:
            if model == 'gemma':
                client = genai.Client(api_key=google_api_key)

                if schema:
                    prompt = (
                        "Return ONLY valid JSON with no code fences. "
                        "Do not add backticks. Do not wrap in markdown. "
                        f"Schema:\n{schema.model_json_schema()}\n\n"
                        f"User prompt:\n{prompt}"
                    )

                response = client.models.generate_content(
                    model='models/gemma-3-27b-it',
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )

                text = response.text
            
            elif model == 'nemotron':
                client = OpenAI(
                    base_url = "https://integrate.api.nvidia.com/v1",
                    api_key = nvidia_api_key
                )

                if schema:
                    prompt = (
                        "Return ONLY valid JSON with no code fences. "
                        "Do not add backticks. Do not wrap in markdown. "
                        f"Schema:\n{schema.model_json_schema()}\n\n"
                        f"User prompt:\n{prompt}"
                    )

                response = client.chat.completions.create(
                    model="nvidia/nemotron-3-nano-30b-a3b",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                text = response.choices[0].message.content

            break
        except Exception as e:
            error_msg = str(e)
            if not (_is_rate_limit_error(e) or _is_transient_error(e)) or attempt == max_retries - 1:
                latency_ms = (time.time() - start_time) * 1000
                cost_tracker.log_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=error_msg
                )
                raise
            delay = min(8, 0.5 * (2 ** attempt)) + random.uniform(0, 0.3)
            time.sleep(delay)

    if text is None:
        text = ""
    text = text.strip()

    # Log successful call
    output_tokens = _estimate_tokens(text)
    latency_ms = (time.time() - start_time) * 1000
    cost_tracker.log_call(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        success=True,
        error=""
    )

    if schema:
        for attempt in range(3):
            try:
                cleaned_text = text
                if cleaned_text.startswith("```"):
                    parts = cleaned_text.split("```")
                    if len(parts) >= 2:
                        cleaned_text = parts[1].strip()
                    if cleaned_text.startswith("json"):
                        cleaned_text = cleaned_text[4:].strip()
                elif cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3].strip()
                if not cleaned_text:
                    return _schema_fallback(schema)
                return schema.model_validate_json(cleaned_text)
            except Exception:
                if attempt == 2:
                    try:
                        return schema.model_validate_json(text)
                    except Exception:
                        return _schema_fallback(schema)
                continue

    return text


class TestEmbeddedSchema(BaseModel):
    number: int = Field(description="A single integer.")
    number_name: str = Field(description="Name of the number.")

class TestSchema(BaseModel):
    answer: str = Field(description="The answer.")
    lists: list[TestEmbeddedSchema] = Field(description="A list of integers.")

def load_prompts(path: str) -> dict[str, str]:
    """Load prompts from a JSON file.
    
    Args:
        path: Path to the JSON file containing prompts

    Returns:
        Dictionary mapping prompt names to prompt text
    """
    with open(path, 'r') as f:
        prompts: dict[str, str] = json.load(f)
    return prompts


def reload_prompts(path: str = "prompts.json") -> dict[str, str]:
    """Reload prompts from disk (for hot-reloading during development).
    
    Args:
        path: Path to the JSON file containing prompts
        
    Returns:
        Fresh dictionary of prompts loaded from disk
    """
    return load_prompts(path)