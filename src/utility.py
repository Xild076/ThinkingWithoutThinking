from typing import Any, Mapping, Literal
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
import google.genai as genai
from openai import OpenAI
import dotenv
import os
import json
import time
import random

Schema = genai.types.Schema

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


def _default_for_type(annotation):
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


def _schema_fallback(schema: BaseModel):
    values = {}
    for name, field in schema.model_fields.items():
        if field.default is not PydanticUndefined:
            values[name] = field.default
        else:
            values[name] = _default_for_type(field.annotation)
    return schema.model_validate(values)


def generate_text(prompt, model:Literal['gemma', 'nemotron']='gemma', schema: BaseModel = None, temperature=0.7, max_tokens=None, max_retries: int = 4, max_prompt_chars: int = 12000):
    """Generate text using Google GenAI models.
    
    Args:
        prompt (str): Input text prompt
        model (str): Model identifier
        schema (BaseModel, optional): Pydantic model for response validation
        temperature (float): Sampling temperature
        max_tokens (int, optional): Maximum output tokens

    Returns:
        str or BaseModel: Generated text or validated response
    """
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]

    if model == 'gemma':
        temperature = max(0.0, min(2.0, temperature))
    elif model == 'nemotron':
        temperature = max(0.0, min(1.0, temperature))
    
    text = None
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
            if not (_is_rate_limit_error(e) or _is_transient_error(e)) or attempt == max_retries - 1:
                raise
            delay = min(8, 0.5 * (2 ** attempt)) + random.uniform(0, 0.3)
            time.sleep(delay)


    if text is None:
        text = ""
    text = text.strip()

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

def load_prompts(path: str):
    """Load prompts from a JSON file.
    
    Args:
        path (str): Path to the JSON file containing prompts

    Returns:
        dict: Dictionary of prompts
    """
    with open(path, 'r') as f:
        prompts = json.load(f)
    return prompts

"""
output = generate_text("What is the capital of France? Give me 5 random numbers.", schema=TestSchema)
print(output)
print(type(output))
print(output.answer)
print(output.lists)
"""

"""output = generate_text("what is the capital of France?", model='nemotron', temperature=-1.0)
print(output)"""