from math import floor
from pydoc import text
from typing import Any, Literal, Type, TypeVar, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import PydanticUndefined
try:
    import google.genai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
try:
    import dotenv
except Exception:  # pragma: no cover - optional dependency
    dotenv = None
import os
import json
import time
import random
import re
import logging
from pathlib import Path
try:
    from nltk import sent_tokenize, word_tokenize
except Exception:  # pragma: no cover - optional dependency
    sent_tokenize = None
    word_tokenize = None
try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None

if dotenv is not None:
    dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

# Configure logging
def setup_logging(log_file: str = "logs/pipeline.log") -> logging.Logger:
    """Set up file-based logging for the pipeline
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Only add handler once
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    return stripped


def _repair_json_string(text: str) -> str:
    """Best-effort cleanup for common JSON issues from LLM output."""
    if not text:
        return ""
    cleaned = text
    cleaned = cleaned.replace("\\'", "'")
    cleaned = re.sub(r"\\(?![""\\/bfnrtu])", "", cleaned)
    cleaned = cleaned.replace("\r\n", "\n")
    return cleaned


def _coerce_single_field_schema(text: str, schema: Type[BaseModel]) -> BaseModel | None:
    field_names = list(schema.model_fields.keys())
    if len(field_names) != 1:
        return None
    coerced_payload = {field_names[0]: text.strip()}
    try:
        return schema.model_validate(coerced_payload)
    except Exception:
        return None


def _validation_error_fingerprint(error: ValidationError) -> str:
    try:
        rows: list[str] = []
        for issue in error.errors(include_url=False):
            loc = ".".join(str(part) for part in issue.get("loc", []))
            issue_type = str(issue.get("type", ""))
            message = str(issue.get("msg", ""))
            ctx = issue.get("ctx")
            ctx_text = json.dumps(ctx, sort_keys=True, ensure_ascii=True) if isinstance(ctx, dict) else str(ctx or "")
            rows.append(f"{loc}|{issue_type}|{message}|{ctx_text}")
        rows.sort()
        return " || ".join(rows)
    except Exception:
        return str(error)


def _sanitize_control_char_artifacts(value: Any) -> Any:
    if isinstance(value, str):
        value = value.replace("\t", "\\t")
        value = value.replace("\f", "\\f")
        value = value.replace("\b", "\\b")
        value = value.replace("\r", "")
        return value
    if isinstance(value, list):
        return [_sanitize_control_char_artifacts(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_control_char_artifacts(item) for key, item in value.items()}
    return value


def _extract_numeric_constraints(field: Any) -> tuple[float | None, float | None, float | None, float | None]:
    ge = gt = le = lt = None
    for meta in getattr(field, "metadata", []) or []:
        if ge is None and hasattr(meta, "ge"):
            value = getattr(meta, "ge", None)
            if isinstance(value, (int, float)):
                ge = float(value)
        if gt is None and hasattr(meta, "gt"):
            value = getattr(meta, "gt", None)
            if isinstance(value, (int, float)):
                gt = float(value)
        if le is None and hasattr(meta, "le"):
            value = getattr(meta, "le", None)
            if isinstance(value, (int, float)):
                le = float(value)
        if lt is None and hasattr(meta, "lt"):
            value = getattr(meta, "lt", None)
            if isinstance(value, (int, float)):
                lt = float(value)
    return ge, gt, le, lt


def _fallback_int_for_field(field: Any) -> int:
    ge, gt, le, lt = _extract_numeric_constraints(field)
    value = 1

    if ge is not None:
        value = max(value, int(ge))
    if gt is not None:
        value = max(value, int(gt) + 1)
    if le is not None:
        value = min(value, int(le))
    if lt is not None and value >= int(lt):
        value = int(lt) - 1

    if ge is not None and value < ge:
        value = int(ge)
    if gt is not None and value <= gt:
        value = int(gt) + 1
    return value


def _fallback_float_for_field(field: Any) -> float:
    ge, gt, le, lt = _extract_numeric_constraints(field)
    epsilon = 0.01
    value = 1.0

    if ge is not None:
        value = max(value, float(ge))
    if gt is not None:
        value = max(value, float(gt) + epsilon)
    if le is not None:
        value = min(value, float(le))
    if lt is not None and value >= float(lt):
        value = float(lt) - epsilon

    if ge is not None and value < ge:
        value = float(ge)
    if gt is not None and value <= gt:
        value = float(gt) + epsilon
    return value


def _graceful_schema_fallback(schema: Type[BaseModel], message: str) -> BaseModel | None:
    payload: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Literal and args:
            payload[field_name] = args[0]
            continue

        if annotation is str:
            payload[field_name] = message
        elif annotation is bool:
            payload[field_name] = False
        elif annotation is int:
            payload[field_name] = _fallback_int_for_field(field)
        elif annotation is float:
            payload[field_name] = _fallback_float_for_field(field)
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested = _graceful_schema_fallback(annotation, message)
            payload[field_name] = nested.model_dump() if nested is not None else {}
        elif annotation is dict or origin is dict:
            payload[field_name] = {}
        elif annotation is list or origin is list:
            payload[field_name] = []
        else:
            payload[field_name] = message

    try:
        return schema.model_validate(payload)
    except Exception:
        return None


def _extract_retry_after_seconds(error_text: str) -> float | None:
    patterns = [
        r"retry in\s*([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
        r"Please retry in\s*([0-9]+(?:\.[0-9]+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _is_non_retryable_error(error_text: str) -> bool:
    lowered = error_text.lower()
    non_retryable_patterns = [
        "maximum context length",
        "input tokens",
        "context length",
        "too many tokens",
        "request too large",
        "tokens per minute",
        "tpm",
        "json_invalid",
        "unknown model",
        "package is not installed",
        "no module named",
        "groq package is not installed",
        "google-genai package is not installed",
        "openai package is not installed",
    ]
    return any(pattern in lowered for pattern in non_retryable_patterns)


def _is_quota_exhausted(error_text: str) -> bool:
    lowered = error_text.lower()
    patterns = [
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "429",
    ]
    return any(pattern in lowered for pattern in patterns)


def _request_timeout_seconds(
    model: str,
    prompt: str,
    schema: Type[BaseModel] | None,
) -> float:
    base = 45.0
    if model == "nemotron":
        base = 60.0
    elif model == "oss120b":
        base = 70.0

    prompt_len = len(prompt or "")
    if prompt_len > 12000:
        base += 20.0
    elif prompt_len > 7000:
        base += 10.0

    if schema is not None:
        base += 8.0

    return min(140.0, max(30.0, base))

TEMPERATURE_LIMITS: dict[str, tuple[float, float]] = {
    "gemma": (0.0, 2.0),
    "nemotron": (0.0, 1.0),
    "oss120b": (0.0, 1.0)
}


# Prompt loading utilities

LOADED_PROMPTS = {
    "path": "",
    "content": {}
}


def _resolve_existing_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    root_dir = Path(__file__).resolve().parents[1]
    search_candidates = [
        root_dir / path,
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
    ]

    for resolved in search_candidates:
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Prompt file not found: {path}. Checked: "
        + ", ".join(str(p) for p in [candidate, *search_candidates])
    )

def load_prompts(path: str) -> dict:
    """Loads a prompt from a text file

    Args:
        path (str): Path to the prompt file
    
    Returns:
        dict: A dictionary containing the prompt ids (e.g., initial_plan_creation_block, sub_plan_creation_block, etc) as keys and the corresponding prompt content as values
    """
    global LOADED_PROMPTS
    resolved_path = _resolve_existing_path(path)
    resolved_path_str = str(resolved_path)
    if LOADED_PROMPTS["path"] == resolved_path_str:
        return LOADED_PROMPTS

    with resolved_path.open("r", encoding="utf-8") as handle:
        LOADED_PROMPTS['content'] = json.load(handle)
    LOADED_PROMPTS["path"] = resolved_path_str
    return LOADED_PROMPTS

def get_prompt(prompt_id: str) -> str:
    """Gets a specific prompt from the loaded prompts

    Args:
        prompt_id (str): The id of the prompt to retrieve (e.g., initial_plan_creation_block, sub_plan_creation_block, etc)
    Returns:
        str: The content of the requested prompt
    """
    if prompt_id in LOADED_PROMPTS['content']:
        return LOADED_PROMPTS['content'][prompt_id]
    else:
        raise ValueError(f"Prompt id '{prompt_id}' not found in loaded prompts.")

def update_prompt(section: str, new_content: str) -> None:
    """Updates a specific section of the loaded prompt

    Args:
        section (str): The section to update (e.g., initial_plan_creation_block, sub_plan_creation_block, etc)
        new_content (str): The new content to replace the existing section with
    """
    global LOADED_PROMPTS
    if section in LOADED_PROMPTS['content']:
        LOADED_PROMPTS['content'][section] = new_content
        with open(LOADED_PROMPTS["path"], "w") as f:
            json.dump(LOADED_PROMPTS['content'], f, indent=4)
    else:
        raise ValueError(f"Section '{section}' not found in loaded prompts.")

def save_prompt() -> None:
    """Saves the currently loaded prompt to a text file"""
    with open(LOADED_PROMPTS["path"], "w") as f:
        json.dump(LOADED_PROMPTS['content'], f, indent=4)

# LLM-related utilities and helper functions

def _clamp_temperature(model: str, temperature: float) -> float:
    """Clamps temperature depending on specific model parameters

    Args:
        model (str): model name (gemma, nemotron, etc)
        temperature (float): temperature value to clamp

    Returns:
        float: clamped temperature value
    """
    min_temp, max_temp = TEMPERATURE_LIMITS.get(model, (0.0, 1.0))
    try:
        temp_value = float(temperature)
    except Exception:
        temp_value = min_temp
    return max(min_temp, min(max_temp, temp_value))

def generate_text(
    prompt: str,
    model: Literal["gemma", "nemotron", "oss120b"] = "gemma",
    schema: Type[BaseModel] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    retries: int = 12,
    retry_delay: float = 1.5,
    max_total_retry_wait: float = 120.0):
    """Generates text through LLM API calls

    Args:
        prompt (str): The prompt to send to the model
        model (Literal["gemma", "nemotron", "oss120b"], optional): The model to use. Defaults to "gemma".
        schema (Type[BaseModel] | None, optional): The Pydantic schema to validate the output against. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to None.

    Raises:
        ValueError: If the model output cannot be parsed into the provided schema.

    Returns:
        str | BaseModel: The generated text or the parsed schema instance
    """
    prompt_preview = prompt[:100].replace('\n', ' ')
    logger.info(f"generate_text() called - model={model}, schema={schema.__name__ if schema else 'None'}, temp={temperature}, prompt_start: {prompt_preview}...")

    base_temperature = temperature
    temperature = _clamp_temperature(model, temperature)

    if schema:
        prompt = (
            "Return ONLY valid JSON with no code fences. "
            "Do not add backticks. Do not wrap in markdown. "
            f"Schema:\n{schema.model_json_schema()}\n\n"
            f"User prompt:\n{prompt}"
        )

    last_error = None
    total_waited = 0.0
    switched_model = False
    fallback_model = "nemotron"
    request_timeout = _request_timeout_seconds(model, prompt, schema)
    repeated_validation_fingerprint = None
    repeated_validation_count = 0
    validation_retry_cap = 3
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Attempt {attempt}/{retries} for {model} (timeout={request_timeout:.1f}s)")
            
            if model == "gemma":
                if genai is None:
                    raise RuntimeError("google-genai package is not installed")
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
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
                if OpenAI is None:
                    raise RuntimeError("openai package is not installed")
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    timeout=request_timeout,
                    max_retries=0,
                )

                response = client.chat.completions.create(
                    model="nvidia/nemotron-3-nano-30b-a3b",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                text = response.choices[0].message.content
            elif model == 'oss120b':
                if OpenAI is None:
                    raise RuntimeError("openai package is not installed")
                client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY"),
                    timeout=request_timeout,
                    max_retries=0,
                )

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                text = response.choices[0].message.content
            else:
                raise ValueError(f"Unknown model: {model}")

            text_preview = text[:50].replace('\n', ' ')
            logger.debug(f"API response received: {text_preview}...")

            if schema:
                cleaned_text = _strip_code_fences(text)
                try:
                    parsed = schema.model_validate_json(cleaned_text)
                    parsed_payload = _sanitize_control_char_artifacts(parsed.model_dump())
                    parsed = schema.model_validate(parsed_payload)
                    logger.info(f"Schema validation successful for {schema.__name__}")
                    return parsed
                except ValidationError:
                    repaired_text = _repair_json_string(cleaned_text)
                    if repaired_text != cleaned_text:
                        try:
                            parsed = schema.model_validate_json(repaired_text)
                            parsed_payload = _sanitize_control_char_artifacts(parsed.model_dump())
                            parsed = schema.model_validate(parsed_payload)
                            logger.warning(f"Schema validation recovered via JSON repair for {schema.__name__}")
                            return parsed
                        except ValidationError:
                            pass

                    fallback_parsed = _coerce_single_field_schema(cleaned_text, schema)
                    if fallback_parsed is not None:
                        logger.warning(
                            f"Schema JSON parse failed for {schema.__name__}; used plain-text fallback for single-field schema"
                        )
                        return fallback_parsed
                    raise

            logger.info(f"generate_text() success for {model}")
            return text
        except Exception as e:
            last_error = e
            error_text = str(e)
            logger.warning(f"Attempt {attempt} failed: {type(e).__name__}: {error_text[:150]}")
            if isinstance(e, ValidationError):
                current_fingerprint = _validation_error_fingerprint(e)
                if current_fingerprint == repeated_validation_fingerprint:
                    repeated_validation_count += 1
                else:
                    repeated_validation_fingerprint = current_fingerprint
                    repeated_validation_count = 1

                logger.debug(
                    "Schema validation fingerprint for %s attempt %s/%s: %s",
                    schema.__name__ if schema else "unknown_schema",
                    attempt,
                    retries,
                    current_fingerprint[:240],
                )
                if repeated_validation_count >= validation_retry_cap:
                    logger.warning(
                        "Repeated schema validation mismatch detected (%s consecutive attempts); stopping retries early",
                        repeated_validation_count,
                    )
                    break
            else:
                repeated_validation_fingerprint = None
                repeated_validation_count = 0

            if model == "gemma" and not switched_model and attempt >= 4 and _is_quota_exhausted(error_text):
                switched_model = True
                model = fallback_model
                temperature = _clamp_temperature(model, base_temperature)
                request_timeout = _request_timeout_seconds(model, prompt, schema)
                logger.warning("Quota exhaustion detected; switching to nemotron for remaining retries")
                continue
            if _is_non_retryable_error(error_text):
                logger.warning("Encountered non-retryable error; skipping remaining retries and degrading gracefully")
                break
            if attempt == retries:
                break
            remaining_wait_budget = max_total_retry_wait - total_waited
            if remaining_wait_budget <= 0:
                logger.warning("Retry wait budget exhausted; moving to graceful degradation")
                break

            exponential_wait = retry_delay * (2 ** (attempt - 1))
            provider_retry_after = _extract_retry_after_seconds(error_text)
            sleep_for = max(exponential_wait, provider_retry_after or 0.0) + random.uniform(0, 0.4)
            sleep_for = min(sleep_for, remaining_wait_budget)

            logger.debug(f"Retrying in {sleep_for:.2f} seconds...")
            time.sleep(sleep_for)
            total_waited += sleep_for

    error_msg = f"Failed to generate text after {retries} attempts: {last_error}"
    logger.error(error_msg)

    degraded_message = (
        "[Graceful degradation] Upstream model call failed after retries. "
        "Returning best-effort fallback content."
    )

    if schema:
        fallback_parsed = _coerce_single_field_schema(degraded_message, schema)
        if fallback_parsed is not None:
            logger.warning(f"Returning single-field schema fallback for {schema.__name__}")
            return fallback_parsed

        fallback_schema = _graceful_schema_fallback(schema, degraded_message)
        if fallback_schema is not None:
            logger.warning(f"Returning multi-field schema fallback for {schema.__name__}")
            return fallback_schema

    return degraded_message


# Reading level utilities

def _safe_word_tokenize(text: str) -> list[str]:
    if word_tokenize is None:
        return [token for token in re.findall(r"[a-zA-Z0-9']+", text or "")]
    try:
        return word_tokenize(text)
    except Exception:
        return [token for token in re.findall(r"[a-zA-Z0-9']+", text or "")]


def _safe_sent_tokenize(text: str) -> list[str]:
    if sent_tokenize is None:
        return [part.strip() for part in re.split(r"(?<=[.!?])\\s+", text or "") if part.strip()]
    try:
        return sent_tokenize(text)
    except Exception:
        return [part.strip() for part in re.split(r"(?<=[.!?])\\s+", text or "") if part.strip()]


def _load_dale_chell() -> dict[str, Any]:
    with open("data/dale_chell.txt", "r") as f:
        data = f.read()
    words = data.split("\n")
    return words

def sylco(word: str) -> int:
    """Finds the syllable count of a word

    Args:
        word (str): Input word

    Returns:
        int: Number of syllables in the word
    
    Courtesy of https://stackoverflow.com/questions/46759492/syllable-count-in-python
    """
    word = word.lower()

    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']

    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']

    pre_one = ['preach']

    syls = 0
    disc = 0

    if len(word) <= 3 :
        syls = 1
        return syls


    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1


    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass

        else :
            disc+=1


    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple

    numVowels = len(re.findall(r'[eaoui]',word))

    if word[:2] == "mc" :
        syls+=1

    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1


    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1


    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1

    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1


    if word[-3:] == "ian" : 
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1


    if word[:2] == "co" and word[2] in 'eaoui' :

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1


    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1


    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass   


    if word in exception_del :
        disc+=1

    if word in exception_add :
        syls+=1     

    return numVowels - disc + syls


# Reading level calculations

def dale_chell_reading_level(text: str) -> int:
    """Generates reading level based on Dale-Chell score.

    Args:
        text (str): Input text

    Returns:
        int: Grade level of the text (0-13)
    """
    words = _safe_word_tokenize(text)
    sentences = _safe_sent_tokenize(text)
    num_words = len(words)
    num_difficult_words = num_words - sum(1 for word in words if word.lower() in _load_dale_chell())
    num_sentences = len(sentences)

    dale_chell = 0.1579 * (num_difficult_words / num_words) * 100 + 0.0496 * (num_words / num_sentences) + 3.6365

    return int(floor(max(0, min(13, dale_chell))))

def forcast_reading_level(text: str) -> int:
    """Generate reading level based on Forcast score.

    Args:
        text (str): Input text

    Returns:
        int: Grade level of the text (0-13)
    """
    words = _safe_word_tokenize(text)
    num_1_syllable_words = sum(1 for word in words if sylco(word) == 1)
    num_words = len(words)
    forcast = 20 - (num_1_syllable_words / num_words) * 15

    return int(floor(max(0, min(13, forcast))))

def gunning_fog_reading_level(text: str) -> int:
    """Generates reading level based on gunning-fog score.

    Args:
        text (str): Input text

    Returns:
        int: Grade level of the text (0-13)
    """
    words = _safe_word_tokenize(text)
    sentences = _safe_sent_tokenize(text)
    num_words = len(words)
    num_complex_words = sum(1 for word in words if sylco(word) >= 3)
    num_sentences = len(sentences)

    gunning_fog = 0.4 * ((num_words / num_sentences) + 100 * (num_complex_words / num_words))

    return int(floor(max(0, min(13, gunning_fog))))

def flesch_kincaid_reading_level(text: str) -> int:
    """Generates reading level based on flesch-kincaid score.

    Args:
        text (str): Input text

    Returns:
        int: Grade level of the text (0-13)
    """
    words = _safe_word_tokenize(text)
    sentences = _safe_sent_tokenize(text)
    num_words = len(words)
    num_syllables = sum(sylco(word) for word in words)
    num_sentences = len(sentences)

    flesch_kincaid = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    return int(floor(max(0, min(13, flesch_kincaid))))

def get_aggregate_reading_level(text: str, weights: dict={'dale_chell': 0.35, 'forcast': 0.15, 'gunning_fog': 0.35, 'flesch_kincaid': 0.15}) -> int:
    """Generates aggregate reading level with higher weight on higher accuracy measurements (Dale-Chell, Gunning-Fog)

    Args:
        text (str): Input text
        weights (dict, optional): Weights for each reading level measurement. Defaults to {'dale_chell': 0.35, 'forcast': 0.15, 'gunning_fog': 0.35, 'flesch_kincaid': 0.15}.

    Returns:
        int: Grade level of the text (0-13)
    """
    dale_chell = dale_chell_reading_level(text)
    forcast = forcast_reading_level(text)
    gunning_fog = gunning_fog_reading_level(text)
    flesch_kincaid = flesch_kincaid_reading_level(text)

    aggregate = dale_chell * weights['dale_chell'] + forcast * weights['forcast'] + gunning_fog * weights['gunning_fog'] + flesch_kincaid * weights['flesch_kincaid']

    return int(round(max(0, min(13, aggregate))))
