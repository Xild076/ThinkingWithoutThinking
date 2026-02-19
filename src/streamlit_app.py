from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

try:
    from pipeline import Pipeline
except Exception:  # pragma: no cover
    from src.pipeline import Pipeline


st.set_page_config(
    page_title="ThinkingWithoutThinking",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(1200px 500px at 15% -10%, rgba(45, 120, 220, 0.18), transparent),
          radial-gradient(900px 500px at 100% 0%, rgba(26, 188, 156, 0.16), transparent),
          #0f1420;
      }
      .twt-hero {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        background: linear-gradient(140deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
      }
      .twt-hero h1 {
        margin: 0;
        color: #e9f0ff;
        font-size: 1.45rem;
      }
      .twt-hero p {
        margin: 0.35rem 0 0;
        color: #b8c7df;
        font-size: 0.92rem;
      }
      .twt-chip {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        margin-right: 0.4rem;
        font-size: 0.78rem;
        color: #dff3ff;
        background: rgba(92, 164, 255, 0.18);
        border: 1px solid rgba(92, 164, 255, 0.35);
      }
      .twt-section-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        background: rgba(18, 26, 40, 0.72);
        padding: 0.8rem 0.95rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="twt-hero">
      <h1>ThinkingWithoutThinking · Mission Console</h1>
      <p>Live event trace, final synthesis, critique signals, and concrete improvement guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_events" not in st.session_state:
    st.session_state.last_events = []


def _read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass

    for section in ("api_keys", "api", "keys"):
        try:
            block = st.secrets.get(section)
            if isinstance(block, dict):
                nested = block.get(name)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        except Exception:
            continue
    return None


def _bootstrap_env_from_secrets() -> dict[str, bool]:
    key_names = ["GOOGLE_API_KEY", "NVIDIA_API_KEY", "GROQ_API_KEY"]
    loaded: dict[str, bool] = {}
    for key_name in key_names:
        secret_val = _read_secret(key_name)
        if secret_val:
            os.environ[key_name] = secret_val
            loaded[key_name] = True
        else:
            loaded[key_name] = bool(os.getenv(key_name))
    return loaded


secrets_status = _bootstrap_env_from_secrets()

pipeline = Pipeline()


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "message", "detail", "reason"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        out.append(candidate.strip())
                        break
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key, val in value.items():
            if isinstance(val, str) and val.strip():
                out.append(f"{key}: {val.strip()}")
        return out
    return []


def _extract_feedback(result: dict[str, Any] | None, events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    critique_keys = ["critique", "critiques", "limitations", "weaknesses", "caveats", "risks", "issues"]
    improve_keys = ["improvements", "improvement", "suggestions", "next_steps", "action_items", "recommendations"]

    critiques: list[str] = []
    improvements: list[str] = []

    payload = result if isinstance(result, dict) else {}
    for key in critique_keys:
        critiques.extend(_to_list(payload.get(key)))
    for key in improve_keys:
        improvements.extend(_to_list(payload.get(key)))

    for event in events:
        event_type = str(event.get("event_type") or "")
        stage = str(event.get("stage") or "")
        event_payload = event.get("payload")

        if event_type in {"warning", "error"}:
            message = None
            if isinstance(event_payload, dict):
                message = event_payload.get("message") or event_payload.get("error")
            if isinstance(message, str) and message.strip():
                critiques.append(f"{event_type.upper()} ({stage}): {message.strip()}")

        if stage in {"improvement", "improvements"} and isinstance(event_payload, dict):
            for key in improve_keys:
                improvements.extend(_to_list(event_payload.get(key)))

    def _dedupe(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            key = line.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(line)
        return out

    critiques = _dedupe(critiques)
    improvements = _dedupe(improvements)

    return critiques, improvements


def _event_stats(events: list[dict[str, Any]]) -> dict[str, int]:
    warnings = sum(1 for event in events if event.get("event_type") == "warning")
    errors = sum(1 for event in events if event.get("event_type") == "error")
    finals = sum(1 for event in events if event.get("event_type") == "final_result")
    return {
        "events": len(events),
        "warnings": warnings,
        "errors": errors,
        "final_results": finals,
    }

with st.sidebar:
    st.markdown("### Run Controls")
    prompt = st.text_area("Prompt", height=220, placeholder="Enter a prompt...")
    thinking_level = st.selectbox(
        "Thinking Level",
        options=["low", "med-synth", "med-plan", "high"],
        index=1,
    )
    run_button = st.button("Run")
    clear_button = st.button("Clear")

    missing = [name for name, ok in secrets_status.items() if not ok]
    if missing:
        st.caption("Missing secrets: " + ", ".join(missing))
        st.caption("Add them to .streamlit/secrets.toml")

if clear_button:
    st.session_state.last_result = None
    st.session_state.last_events = []
    st.rerun()


if run_button:
    if not prompt.strip():
        st.error("Prompt cannot be empty")
    else:
        progress = st.progress(0)
        status = st.empty()
        event_container = st.container()

        all_events: list[dict] = []
        final_payload = None

        with event_container:
            for idx, event in enumerate(
                pipeline.run_stream(prompt=prompt, thinking_level=thinking_level)
            ):
                all_events.append(event)
                percent = min(1.0, 0.05 + (idx * 0.04))
                progress.progress(percent)
                status.caption(
                    f"{event.get('event_type')} | stage={event.get('stage')} | eta={event.get('eta_seconds')}"
                )

                with st.expander(
                    f"{idx + 1:02d}. {event.get('event_type')} ({event.get('stage')})",
                    expanded=idx < 3 or event.get("event_type") in {"warning", "error", "complete"},
                ):
                    st.code(json.dumps(event.get("payload", {}), indent=2), language="json")

                if event.get("event_type") == "final_result":
                    final_payload = event.get("payload")

        progress.progress(1.0)
        status.caption("Complete")

        st.session_state.last_result = final_payload
        st.session_state.last_events = all_events
        st.session_state.history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt,
                "thinking_level": thinking_level,
                "events": all_events,
                "result": final_payload,
            }
        )


if st.session_state.last_events:
    stats = _event_stats(st.session_state.last_events)
    st.markdown(
        (
            f"<span class='twt-chip'>events: {stats['events']}</span>"
            f"<span class='twt-chip'>warnings: {stats['warnings']}</span>"
            f"<span class='twt-chip'>errors: {stats['errors']}</span>"
            f"<span class='twt-chip'>final payloads: {stats['final_results']}</span>"
        ),
        unsafe_allow_html=True,
    )


if st.session_state.last_result:
    critiques, improvements = _extract_feedback(st.session_state.last_result, st.session_state.last_events)

    tab_response, tab_feedback, tab_payload = st.tabs(["Response", "Critique & Improvements", "Payload"])

    with tab_response:
        st.markdown("### Final Response")
        st.markdown(st.session_state.last_result.get("response", ""))

        embeddings = st.session_state.last_result.get("image_embeddings", [])
        if isinstance(embeddings, list) and embeddings:
            st.markdown("### Embedded Images")
            for embed in embeddings:
                data_uri = str(embed.get("data_uri", ""))
                if not data_uri.startswith("data:") or "," not in data_uri:
                    continue
                _, encoded = data_uri.split(",", 1)
                try:
                    image_bytes = base64.b64decode(encoded)
                    caption = str(embed.get("path", "embedded image"))
                    st.image(image_bytes, caption=caption, use_container_width=True)
                except Exception:
                    continue

    with tab_feedback:
        critique_col, improve_col = st.columns(2)
        with critique_col:
            st.markdown("### Critique")
            if critiques:
                for item in critiques:
                    st.markdown(f"- {item}")
            else:
                st.info("No explicit critique captured in this run.")

        with improve_col:
            st.markdown("### Improvements")
            if improvements:
                for item in improvements:
                    st.markdown(f"- {item}")
            else:
                st.info("No explicit improvements captured in this run.")

    with tab_payload:
        st.markdown("### Final Payload")
        st.code(json.dumps(st.session_state.last_result, indent=2), language="json")

if st.session_state.history:
    with st.expander("Run History"):
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(
                f"**{item['timestamp']}** | `{item['thinking_level']}` | {item['prompt'][:80]}"
            )
