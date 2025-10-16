import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.thinking_pipeline import ThinkingPipeline  # noqa: E402
from src import utility  # noqa: E402


st.set_page_config(
    page_title="Thinking Without Thinking",
    page_icon="🧠",
    layout="wide",
)

# File to store usage data
USAGE_FILE = ROOT_DIR / ".usage_data.json"

def _load_usage_data() -> dict:
    """Load usage tracking data."""
    # We persist only whether a custom API key was configured across server
    # restarts. The free 'uses' counter is intentionally NOT persisted so it
    # resets whenever a new instance or session starts.
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
                return {"has_custom_key": bool(data.get("has_custom_key", False))}
        except Exception:
            pass
    return {"has_custom_key": False}

def _save_usage_data(data: dict):
    """Save usage tracking data."""
    try:
        # Only persist the presence of a custom API key across instances.
        to_write = {"has_custom_key": bool(data.get("has_custom_key", False))}
        with open(USAGE_FILE, 'w') as f:
            json.dump(to_write, f)
    except:
        pass


def _init_state() -> None:
    st.session_state.setdefault("conversations", [])
    st.session_state.setdefault("selected_blocks", {})
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("verbose", True)
    st.session_state.setdefault("current_progress", None)
    # Show info popup on first visit only (per session). The popup can be
    # re-opened by the user via the 'Show Info' button in the sidebar.
    st.session_state.setdefault("show_info_popup", True)
    # Track whether we've already shown the popup in this session so we
    # don't display it repeatedly on reruns.
    st.session_state.setdefault("info_popup_shown", False)

    # Load persisted settings (only has_custom_key). Initialize 'uses' to 0
    # so free uses reset when a new instance or session starts.
    persisted = _load_usage_data()
    st.session_state.setdefault("usage_data", {"uses": 0, "has_custom_key": persisted.get("has_custom_key", False)})
    st.session_state.setdefault("custom_api_key", "")
    st.session_state.setdefault("settings", {
        "creative_temperature": 1.2,
        "synthesis_temperature": 0.6,
        "max_pipeline_blocks": 7,
    })


def _render_payload(payload: Any) -> None:
    if payload is None:
        st.info("No data available for this block.")
        return

    if isinstance(payload, dict):
        structured = payload.get("structured_output")
        if structured:
            st.markdown("**Structured Output**")
            st.json(structured)

        output_text = payload.get("output")
        if output_text:
            st.markdown("**Standard Output**")
            st.code(output_text, language="text")

        final_code = payload.get("final_code")
        if final_code:
            st.markdown("**Final Code**")
            st.code(final_code, language="python")

        attempts = payload.get("attempts")
        if attempts:
            st.markdown("**Attempts**")
            for attempt in attempts:
                with st.expander(
                    f"Attempt {attempt['attempt']} - {'✅' if attempt['success'] else '❌'}",
                    expanded=False,
                ):
                    st.write(
                        {
                            k: v
                            for k, v in attempt.items()
                            if k not in {"raw_code", "stdout"}
                        }
                    )
                    if attempt.get("raw_code"):
                        st.markdown("*Generated Code*")
                        st.code(attempt["raw_code"], language="python")
                    if attempt.get("stdout"):
                        st.markdown("*Captured Output*")
                        st.code(attempt["stdout"], language="text")

        remaining = {
            k: v
            for k, v in payload.items()
            if k
            not in {
                "structured_output",
                "output",
                "final_code",
                "attempts",
                "files",
                "plots",
            }
        }
        if remaining:
            st.markdown("**Additional Fields**")
            # Check if it's ideas from creative block - render nicely
            if "ideas" in remaining:
                ideas = remaining["ideas"]
                if isinstance(ideas, str):
                    st.markdown(ideas)
                else:
                    st.write(ideas)
                # Remove ideas from remaining
                remaining = {k: v for k, v in remaining.items() if k != "ideas"}
            
            # Show any other remaining fields as JSON
            if remaining:
                st.json(remaining)

        assets = payload.get("plots") or payload.get("files") or []
        if assets:
            st.markdown("**Generated Assets**")
            for asset in assets:
                st.write(asset)
                try:
                    st.image(asset)
                except Exception:
                    st.warning(f"Could not display image at {asset}")
    else:
        st.write(payload)

def _render_answer_with_images(answer: str, assets: List[str]) -> None:
    """Render final answer text with embedded images from markdown or asset paths."""
    import re
    import os
    
    # Add custom CSS for better answer styling
    st.markdown("""
        <style>
        .answer-container {
            background: transparent;
            border-radius: 0;
            padding: 0;
            margin: 0;
        }
        .answer-content {
            background: transparent;
            border-radius: 0;
            padding: 0;
            line-height: 1.8;
            font-size: 1.05rem;
        }
        .answer-content p {
            margin-bottom: 1.2em;
        }
        .answer-content h1, .answer-content h2, .answer-content h3 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }
        /* Make images smaller and centered */
        .answer-content img, .stImage img {
            max-width: 400px !important;
            width: 100% !important;
            height: auto !important;
            display: block;
            margin: 1.5em auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # CRITICAL: Filter out any markdown image references that don't exist on disk
    # Pattern matches: ![description](path) or ![](path)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def validate_and_clean_answer(text: str) -> str:
        """Remove image references for files that don't exist."""
        matches = list(re.finditer(image_pattern, text))
        if not matches:
            return text
        
        # Process in reverse to maintain indices
        cleaned = text
        for match in reversed(matches):
            image_path = match.group(2)
            if not os.path.isfile(image_path):
                # Remove the entire image markdown including surrounding whitespace
                start = match.start()
                end = match.end()
                # Remove trailing newlines after the image reference
                while end < len(cleaned) and cleaned[end] in '\n\r':
                    end += 1
                cleaned = cleaned[:start] + cleaned[end:]
        
        return cleaned
    
    # Clean the answer of non-existent images
    answer = validate_and_clean_answer(answer)
    
    # Find all remaining valid image references in the answer
    matches = list(re.finditer(image_pattern, answer))
    
    if not matches and not assets:
        # No images, just render the text with styling - enable HTML for superscripts
        st.markdown('<div class="answer-container"><div class="answer-content">', unsafe_allow_html=True)
        st.markdown(answer, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        return
    
    # Render text with images in styled container
    st.markdown('<div class="answer-container"><div class="answer-content">', unsafe_allow_html=True)
    
    last_end = 0
    for match in matches:
        # Render text before the image - enable HTML for superscripts
        if match.start() > last_end:
            st.markdown(answer[last_end:match.start()], unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close content div for image
        
        # Render the image
        image_path = match.group(2)
        image_desc = match.group(1) or "Generated visualization"
        
        try:
            st.image(image_path, caption=image_desc, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display image: {image_path}")
        
        st.markdown('<div class="answer-content">', unsafe_allow_html=True)  # Reopen content div
        last_end = match.end()
    
    # Render any remaining text after the last image - enable HTML for superscripts
    if last_end < len(answer):
        st.markdown(answer[last_end:], unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)  # Close both divs
    
    # If no images were embedded in markdown but we have assets, show them
    if not matches and assets:
        st.markdown("---")
        st.markdown("### 📊 Generated Visualizations")
        for asset_path in assets:
            try:
                st.image(asset_path, use_container_width=True)
            except Exception:
                st.warning(f"Could not display image: {asset_path}")


def _build_conversation_context(current_prompt: str, conversations: List[Dict[str, Any]]) -> str:
    """
    Build intelligent conversation context by extracting relevant information
    from prior conversations based on the current prompt.
    
    Uses a multi-strategy approach:
    1. Keyword overlap to find topically related prior exchanges
    2. Recency weighting (recent conversations get priority)
    3. Entity/fact extraction from answers
    4. Token budget management
    """
    import re
    
    if not conversations:
        return ""
    
    # Extract keywords from current prompt (simple tokenization)
    current_keywords = set(re.findall(r'\b\w{4,}\b', current_prompt.lower()))
    
    # Score each conversation by relevance
    scored_convs = []
    for idx, conv in enumerate(conversations):
        user_prompt = conv.get("prompt", "")
        result = conv.get("result", {})
        answer = result.get("final_answer", "")
        
        # Calculate keyword overlap score
        prompt_keywords = set(re.findall(r'\b\w{4,}\b', user_prompt.lower()))
        answer_keywords = set(re.findall(r'\b\w{4,}\b', answer.lower()[:2000]))  # Sample answer
        
        overlap_score = len(current_keywords & (prompt_keywords | answer_keywords))
        
        # Recency bonus (more recent = higher score)
        recency_bonus = (idx + 1) / len(conversations)
        
        total_score = overlap_score + (recency_bonus * 2)
        
        if total_score > 0:
            scored_convs.append({
                "score": total_score,
                "idx": idx,
                "prompt": user_prompt,
                "answer": answer,
                "is_recent": idx >= len(conversations) - 2  # Last 2 conversations
            })
    
    # Sort by score (highest first)
    scored_convs.sort(key=lambda x: x["score"], reverse=True)
    
    # Build context summary with token budget
    MAX_CONTEXT_TOKENS = 2000  # ~500 words, conservative
    context_parts = []
    current_tokens = 0
    
    for conv_data in scored_convs[:3]:  # Max 3 most relevant conversations
        # Extract key facts from the answer
        answer = conv_data["answer"]
        
        # Clean the answer
        cleaned = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', answer)  # Remove images
        cleaned = re.sub(r'```[\s\S]*?```', '[code block]', cleaned)  # Replace code
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        # Extract just the most important sentences (those with numbers, definitions, or conclusions)
        sentences = re.split(r'[.!?]\s+', cleaned)
        key_sentences = []
        
        for sent in sentences:
            # Prioritize sentences with numbers, equations, definitions, or "therefore"/"thus"
            if any(keyword in sent.lower() for keyword in ['=', 'therefore', 'thus', 'is defined', 'result', ':', 'answer']):
                key_sentences.append(sent.strip())
            elif re.search(r'\d+', sent):  # Contains numbers
                key_sentences.append(sent.strip())
        
        # If no key sentences, take first 2 sentences
        if not key_sentences:
            key_sentences = sentences[:2]
        
        # Limit to 3 key sentences
        summary = '. '.join(key_sentences[:3])
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(summary) // 4
        
        if current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS:
            break
        
        context_parts.append(f"Q: {conv_data['prompt'][:100]}...\nKey facts: {summary}")
        current_tokens += estimated_tokens
    
    if not context_parts:
        return ""
    
    return "Relevant information from previous exchanges:\n\n" + "\n\n".join(context_parts)


def _render_progress(conversation_id: str, pipeline_spec: List[Dict[str, Any]], expanded: bool = False) -> Optional[int]:
    """Render thinking progress as a horizontal scrollable progress bar."""
    if not pipeline_spec:
        st.caption("No pipeline blocks recorded.")
        return None

    selected_key = st.session_state["selected_blocks"].get(conversation_id)
    if selected_key is None:
        selected_key = 0
        st.session_state["selected_blocks"][conversation_id] = selected_key

    # Create a container for the progress visualization
    progress_container = st.container()
    
    with progress_container:
        # Header
        st.markdown("**🔄 Reasoning Process**")
        
        # Progress bar visualization
        num_blocks = len(pipeline_spec)
        progress_pct = ((selected_key + 1) / num_blocks) * 100
        st.progress(progress_pct / 100, text=f"Step {selected_key + 1} of {num_blocks}")
        
        # Horizontal block selector with enhanced custom styling
        st.markdown("""
            <style>
            .thinking-progress {
                display: flex;
                gap: 10px;
                padding: 15px 0;
                overflow-x: auto;
                white-space: nowrap;
            }
            .thinking-step {
                min-width: 140px;
                padding: 12px 16px;
                border-radius: 12px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid #e0e0e0;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .thinking-step:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .thinking-step.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-color: #667eea;
                box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
                transform: scale(1.05);
            }
            .thinking-step.completed {
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                border-color: #4caf50;
                box-shadow: 0 2px 6px rgba(76, 175, 80, 0.2);
            }
            .thinking-step-name {
                font-weight: 600;
                font-size: 0.9rem;
                margin-top: 4px;
            }
            .thinking-step-icon {
                font-size: 1.5rem;
                margin-bottom: 6px;
                display: block;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Render block buttons in columns with better naming
        cols = st.columns(min(len(pipeline_spec), 6))
        
        # Create friendly names for blocks
        block_icons = {
            "plan_creation": "📋",
            "use_code_tool": "💻",
            "use_internet_tool": "🌐",
            "creative_idea_generator": "💡",
            "synthesize_final_answer": "✨"
        }
        
        block_friendly_names = {
            "plan_creation": "Planning",
            "use_code_tool": "Code Tool",
            "use_internet_tool": "Web Search",
            "creative_idea_generator": "Ideas",
            "synthesize_final_answer": "Synthesis"
        }
        
        for idx, block in enumerate(pipeline_spec):
            col_idx = idx % len(cols)
            is_selected = selected_key == idx
            block_key = block.get("key", "block")
            block_name = block_friendly_names.get(block_key, block_key.replace("_", " ").title())
            block_icon = block_icons.get(block_key, "⚙️")
            
            icon = "🔵" if is_selected else "✅"
            
            with cols[col_idx]:
                if st.button(
                    f"{icon} {block_icon}\n**{block_name}**",
                    key=f"{conversation_id}_step_{idx}",
                    help=f"View {block_name} details",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_blocks"][conversation_id] = idx
                    selected_key = idx
                    st.rerun()
        
        st.caption("💡 Click any step to inspect its output • 🔵 = selected • ✅ = completed")
    
    return selected_key


def _render_conversation(conv: Dict[str, Any]) -> None:
    prompt = conv.get("prompt", "")
    result = conv.get("result", {})
    pipeline_spec = result.get("pipeline") or []
    context = result.get("context") or {}
    assets = result.get("assets") or []

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Collapsible thinking process section
        if pipeline_spec:
            thinking_expanded_key = f"{conv['id']}_thinking_expanded"
            if thinking_expanded_key not in st.session_state:
                st.session_state[thinking_expanded_key] = False
            
            with st.expander("🧠 View Thinking Process", expanded=st.session_state[thinking_expanded_key]):
                # Show execution plan first
                st.markdown("### 🔄 Reasoning Process")
                plan_text = result.get("plan") or "(Plan not available)"
                with st.expander("📋 Execution Plan", expanded=False):
                    st.markdown(plan_text)
                
                st.markdown("---")
                
                selected_idx = _render_progress(conv["id"], pipeline_spec, expanded=st.session_state[thinking_expanded_key])
                
                if selected_idx is not None and 0 <= selected_idx < len(pipeline_spec):
                    block_spec = pipeline_spec[selected_idx]
                    context_key = f"{block_spec['key']}_{selected_idx}"
                    
                    # Enhanced block detail view
                    block_key = block_spec['key']
                    block_icons = {
                        "plan_creation": "📋",
                        "use_code_tool": "💻",
                        "use_internet_tool": "🌐",
                        "creative_idea_generator": "💡",
                        "synthesize_final_answer": "✨"
                    }
                    icon = block_icons.get(block_key, "⚙️")
                    
                    st.markdown(f"### {icon} {block_key.replace('_', ' ').title()}")
                    
                    # Show block data if available
                    if block_spec.get('data'):
                        with st.expander("📊 Block Configuration", expanded=False):
                            st.json(block_spec['data'])
                    
                    _render_payload(context.get(context_key))
                    st.divider()

        # Render final answer with embedded images (no header)
        final_answer = result.get("final_answer") or "No final answer produced."
        _render_answer_with_images(final_answer, assets)

        with st.expander("Execution Plan", expanded=False):
            st.code(result.get("plan") or "(plan missing)")

        with st.expander("Logs", expanded=False):
            for log in result.get("logs") or []:
                st.markdown(f"**[{log.get('stage', 'unknown')}]** {log.get('message')}")
                if log.get("data"):
                    st.json(log["data"])


def _run_pipeline(prompt: str, verbose: bool) -> Dict[str, Any]:
    """Run pipeline with live progress tracking."""
    # Create placeholders for live progress
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    pipeline_steps_placeholder = st.empty()
    
    # Progress tracking state
    current_stage = {"index": 0, "total": 1, "name": "Initializing...", "all_steps": []}
    
    def progress_callback(current: int, total: int, stage_name: str):
        """Update progress display during pipeline execution."""
        current_stage["index"] = current
        current_stage["total"] = total
        current_stage["name"] = stage_name
        
        # Track all steps if we haven't yet
        if len(current_stage["all_steps"]) != total:
            current_stage["all_steps"] = [""] * total
        
        # Update the current step name
        if current < len(current_stage["all_steps"]):
            current_stage["all_steps"][current] = stage_name
        
        # Calculate progress
        progress = (current + 1) / (total + 1) if total > 0 else 0
        stage_display = stage_name.replace("_", " ").title()
        
        # Show current step
        progress_placeholder.progress(progress, text=f"⚙️ Step {current + 1}/{total}: {stage_display}")
        
        # Show all pipeline steps with current highlighted
        steps_display = []
        for idx, step in enumerate(current_stage["all_steps"]):
            step_name = step.replace("_", " ").title() if step else "..."
            if idx == current:
                steps_display.append(f"**→ {step_name}** (current)")
            elif idx < current:
                steps_display.append(f"✅ {step_name}")
            else:
                steps_display.append(f"⏳ {step_name}" if step else "⏳ ...")
        
        pipeline_steps_placeholder.markdown("**Pipeline:**\n" + " | ".join(steps_display))
    
    # Initialize progress
    progress_callback(0, 1, "Starting pipeline")
    
    # Run pipeline with callback
    pipeline = ThinkingPipeline(verbose=verbose, progress_callback=progress_callback)
    result = pipeline(prompt)
    
    # Show completion
    progress_placeholder.progress(1.0, text="✅ Pipeline complete!")
    status_placeholder.caption("All stages finished")
    time.sleep(0.8)
    
    # Clear progress indicators
    progress_placeholder.empty()
    status_placeholder.empty()
    pipeline_steps_placeholder.empty()
    
    return result


_init_state()

# Info popup modal: show only once per session, or when explicitly requested
if st.session_state.get("show_info_popup", True) and not st.session_state.get("info_popup_shown", False):
    @st.dialog("Welcome to Thinking Without Thinking!", width="large")
    def show_info():
        st.markdown("""
        ### What is this?

        **Thinking Without Thinking (TWT)** is an agentic AI reasoning pipeline that breaks down complex queries into multiple thinking steps, similar to how Chain-of-Thought or Tree-of-Thought models work. However, unlike Chain-of-Thought or Tree-of-Thought, this system is made up entirely of prompting and is extremely discrete in its implementation, making it both more structured, accurate, and explainable.

        ### How it works:
        
        1. **Planning**: Creates an execution plan for your query
        2. **Routing**: Chooses which specialized blocks to run based on the plan
        3. **Execution**: Runs specialized blocks (code execution, web search, creative ideation)
        4. **Synthesis**: Combines all insights from each block into a final answer
        5. **Quality Control**: Scores and improves the response

        ### Available Tools:
        - **Code Tool**: Executes Python code, generates plots
        - **Internet Tool**: Searches and analyzes web content
        - **Creative Ideas**: Brainstorms innovative ideas and concepts
        
        ### Usage Limits:
        - **Free tier**: 3 uses (you have {uses_left} remaining)
        - **Custom API key**: Unlimited uses - add your Google AI API key in the sidebar
        
        ### Get an API Key:
        1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
        2. Create a free API key
        3. Enter it in the sidebar settings

        Happy chatting! :D

        ---
        *Built with Google Gemma 27B AI and Streamlit*
        """.format(uses_left=max(0, 3 - st.session_state["usage_data"]["uses"])))
        
        if st.button("Got it! Let's start"):
            # Mark as shown and close the popup for this session
            st.session_state["show_info_popup"] = False
            st.session_state["info_popup_shown"] = True
            st.rerun()

    # Display the dialog exactly once and mark it as shown so reruns don't
    # re-open it automatically.
    show_info()
    st.session_state["info_popup_shown"] = True

st.title("Thinking Without Thinking")
st.caption("Chat with a custom reasoning pipeline that has extensive capabilities including web search and code execution. This is fully runnable on a free Google AI API key.")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key Section
    st.subheader("🔑 API Key")
    usage_data = st.session_state["usage_data"]
    
    if usage_data["has_custom_key"] or st.session_state.get("custom_api_key"):
        st.success("✅ Custom API key active - Unlimited uses!")
    else:
        uses_left = max(0, 3 - usage_data["uses"])
        if uses_left > 0:
            st.info(f"🆓 Free tier: {uses_left} use(s) remaining")
        else:
            st.error("❌ Free uses exhausted. Please add an API key to continue.")
    
    api_key_input = st.text_input(
        "Google AI API Key",
        type="password",
        value=st.session_state.get("custom_api_key", ""),
        help="Get a free key at https://aistudio.google.com/apikey",
        placeholder="Enter your API key..."
    )
    
    if api_key_input != st.session_state.get("custom_api_key", ""):
        st.session_state["custom_api_key"] = api_key_input
        if api_key_input:
            utility.set_api_key(api_key_input)
            usage_data["has_custom_key"] = True
            _save_usage_data(usage_data)
            st.success("API key saved!")
            st.rerun()
        else:
            utility.set_api_key(None)
            usage_data["has_custom_key"] = False
            _save_usage_data(usage_data)
    
    # Set API key on startup if available
    if st.session_state.get("custom_api_key"):
        utility.set_api_key(st.session_state["custom_api_key"])
    
    st.divider()
    
    # Advanced Settings
    with st.expander("🎛️ Advanced Settings"):
        st.session_state.settings["creative_temperature"] = st.slider(
            "Creative Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.settings.get("creative_temperature", 1.2),
            step=0.1,
            help="Higher = more creative ideas (default: 1.2)"
        )
        
        st.session_state.settings["synthesis_temperature"] = st.slider(
            "Synthesis Temperature",
            min_value=0.0,
            max_value=1.5,
            value=st.session_state.settings.get("synthesis_temperature", 0.6),
            step=0.1,
            help="Higher = more varied responses (default: 0.6)"
        )
        
        st.session_state.settings["max_pipeline_blocks"] = st.number_input(
            "Max Pipeline Blocks",
            min_value=3,
            max_value=10,
            value=st.session_state.settings.get("max_pipeline_blocks", 7),
            help="Maximum reasoning steps (default: 7)"
        )
    
    st.divider()
    
    # Controls
    st.subheader("🎮 Controls")
    st.session_state["verbose"] = st.checkbox(
        "Verbose logging",
        value=st.session_state.get("verbose", True),
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.conversations = []
            st.session_state.selected_blocks = {}
            st.session_state.last_error = None
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Show Info", use_container_width=True):
            # Force the info dialog to show once when requested by the user
            st.session_state["show_info_popup"] = True
            st.session_state["info_popup_shown"] = False
            st.rerun()

error = st.session_state.get("last_error")
if error:
    st.error(error)

for conv in st.session_state.conversations:
    _render_conversation(conv)

# If currently processing a prompt, show it first
if st.session_state.get("current_prompt"):
    with st.chat_message("user"):
        # Show the display version (without context) to the user
        display_prompt = st.session_state.get("current_prompt_display", st.session_state["current_prompt"])
        st.markdown(display_prompt)

# Chat input - always visible
prompt_input = st.chat_input("Ask the pipeline...")

if prompt_input:
    # Check usage limits
    usage_data = st.session_state["usage_data"]
    has_api_key = usage_data.get("has_custom_key", False) or st.session_state.get("custom_api_key", "")
    
    if not has_api_key and usage_data.get("uses", 0) >= 3:
        st.error("⚠️ You've used all 3 free queries. Please add your Google AI API key in the sidebar to continue.")
        st.stop()
    
    # Build intelligent conversation context by extracting relevant info from history
    context_summary = _build_conversation_context(prompt_input, st.session_state.get("conversations", []))
    
    if context_summary:
        composed = f"[Conversation Context]\n{context_summary}\n\n[Current Query]\n{prompt_input}"
    else:
        composed = prompt_input

    # Store BOTH the original prompt (for display) and the composed prompt (for processing)
    st.session_state["current_prompt_display"] = prompt_input  # What user sees
    st.session_state["current_prompt"] = composed  # What pipeline processes
    st.session_state["processing_complete"] = False
    st.rerun()

# Process the stored prompt if it exists and we haven't processed it yet
if st.session_state.get("current_prompt") and not st.session_state.get("processing_complete"):
    with st.spinner("Running pipeline..."):
        try:
            result = _run_pipeline(st.session_state["current_prompt"], st.session_state["verbose"])
            
            # Increment usage count only on success (if no custom API key)
            usage_data = st.session_state["usage_data"]
            if not usage_data.get("has_custom_key", False):
                usage_data["uses"] = usage_data.get("uses", 0) + 1
                st.session_state["usage_data"] = usage_data
                _save_usage_data(usage_data)
                
        except Exception as exc:  # pragma: no cover - UI feedback
            st.session_state.last_error = str(exc)
            st.session_state["current_prompt"] = None
            st.session_state["current_prompt_display"] = None  # Clear display version too
            st.session_state["processing_complete"] = True
            st.rerun()
        else:
            conversation = {
                "id": f"conv_{int(time.time() * 1000)}",
                "prompt": st.session_state.get("current_prompt_display", st.session_state["current_prompt"]),  # Store display version
                "result": result,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            st.session_state.conversations.append(conversation)
            st.session_state.selected_blocks[conversation["id"]] = None
            st.session_state.last_error = None
            st.session_state["current_prompt"] = None
            st.session_state["current_prompt_display"] = None  # Clear display version too
            st.session_state["processing_complete"] = True
            st.rerun()
