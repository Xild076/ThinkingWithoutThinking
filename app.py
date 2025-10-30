import sys
import time
import json
import re
import os
import tempfile
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
from src.pdf_utils import extract_text_from_pdf, create_pdf_from_markdown  # noqa: E402


st.set_page_config(
    page_title="Thinking Without Thinking",
    page_icon="🧠",
    layout="wide",
)

# File to store usage data
USAGE_FILE = ROOT_DIR / ".usage_data.json"


def _should_create_pdf(prompt: str, result: Dict[str, Any]) -> bool:
    """Determine if a PDF should be automatically created based on the query and result."""
    prompt_lower = prompt.lower()
    
    # Keywords that suggest user wants downloadable content
    pdf_keywords = [
        "study guide", "guide", "report", "summary", "documentation", "notes",
        "cheat sheet", "reference", "tutorial", "instructions", "manual",
        "worksheet", "handout", "overview", "primer", "syllabus"
    ]
    
    # Keywords that suggest user just wants explanation (no PDF)
    no_pdf_keywords = [
        "explain", "what is", "how does", "why", "difference between",
        "compare", "tell me about", "describe"
    ]
    
    # Check if user explicitly wants a PDF
    if "pdf" in prompt_lower or "download" in prompt_lower:
        return True
    
    # Check for PDF-suggesting keywords
    has_pdf_keyword = any(keyword in prompt_lower for keyword in pdf_keywords)
    has_no_pdf_keyword = any(keyword in prompt_lower for keyword in no_pdf_keywords)
    
    # Don't create PDF for simple explanations
    if has_no_pdf_keyword and not has_pdf_keyword:
        return False
    
    # Create PDF if it's a study/reference material request
    if has_pdf_keyword:
        return True
    
    # Check if result is substantial (long answer = likely wants to save it)
    final_answer = result.get("final_answer", "")
    if len(final_answer) > 2000:  # Long, detailed response
        return True
    
    return False


def _extract_pdf_content_for_pdf(result: Dict[str, Any]) -> str:
    """Extract only the relevant final answer content for PDF creation."""
    final_answer = result.get("final_answer", "")
    
    # Remove any meta-commentary about the process
    # The PDF should only contain the actual content, not "Here's what I found..."
    lines = final_answer.split('\n')
    filtered_lines = []
    
    skip_patterns = [
        r"^(here'?s|this|the following|i'?ve)",
        r"^based on",
        r"^after (analyzing|reviewing|examining)",
    ]
    
    for line in lines:
        line_lower = line.strip().lower()
        should_skip = any(re.match(pattern, line_lower) for pattern in skip_patterns)
        
        # Keep the line if it's not meta-commentary
        if not should_skip or len(line.strip()) > 100:  # Long lines are usually content
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()


def _create_pdf_from_result(prompt: str, result: Dict[str, Any]) -> Optional[str]:
    """Create a PDF from the pipeline result and return the file path."""
    try:
        # Extract clean content
        content = _extract_pdf_content_for_pdf(result)
        
        if not content or len(content) < 50:
            return None
        
        # Generate title from prompt
        title_words = prompt.split()[:8]
        title = ' '.join(title_words)
        if len(prompt.split()) > 8:
            title += "..."
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^\w\s-]', '', title)[:30].strip().replace(' ', '_')
        filename = f"{safe_title}_{timestamp}.pdf"
        
        # Create PDF in temp directory
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, filename)
        
        success = create_pdf_from_markdown(content, output_path, title=title)
        
        if success:
            return output_path
        return None
    
    except Exception as e:
        st.warning(f"Could not create PDF: {str(e)}")
        return None

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

    # API key is per-session only (not persisted across restarts).
    # Each user session is independent and doesn't affect other instances.
    st.session_state.setdefault("custom_api_key", "")
    
    # Usage tracking - free uses counter resets every session
    st.session_state.setdefault("usage_data", {"uses": 0})
    
    st.session_state.setdefault("settings", {
        "creative_temperature": 1.2,
        "synthesis_temperature": 0.6,
        "max_pipeline_blocks": 7,
    })
    
    # PDF upload state
    st.session_state.setdefault("uploaded_pdf", None)
    st.session_state.setdefault("pdf_text", None)
    st.session_state.setdefault("had_pdf_context", False)


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
    pdf_path = conv.get("pdf_path")  # PDF path if one was created

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

        # PDF Download Container (if PDF was created)
        if pdf_path and os.path.exists(pdf_path):
            st.markdown("---")
            with st.container():
                st.markdown("### 📄 PDF Document Available")
                st.caption("This response has been formatted as a downloadable PDF document.")
                
                with open(pdf_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                    filename = os.path.basename(pdf_path)
                    
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                        st.markdown(f"**📄 {filename}**")
                        st.caption(f"📊 Size: {len(pdf_data) / 1024:.1f} KB • 🕐 {datetime.now().strftime('%I:%M %p')}")
                    with col2:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
            st.caption("💡 This response is available as a downloadable PDF")
            st.markdown("---")

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

# Custom CSS for better UI
st.markdown("""
<style>
    /* Maintain comfortable bottom padding for chat input */
    main .block-container {
        padding-bottom: 6rem !important;
    }

    /* Compact PDF context alert */
    .pdf-attachment-card {
        background-color: #2b2b2b;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .pdf-attachment-card .pdf-icon {
        background-color: #ff6b6b;
        border-radius: 8px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .pdf-attachment-card .pdf-details {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key Section
    st.subheader("🔑 API Key")
    usage_data = st.session_state["usage_data"]
    custom_api_key = st.session_state.get("custom_api_key", "").strip()
    
    if custom_api_key:
        st.success("✅ Session API key active - Unlimited uses!")
    else:
        uses_left = max(0, 3 - usage_data.get("uses", 0))
        if uses_left > 0:
            st.info(f"🆓 Free tier: {uses_left} use(s) remaining (or use secret API)")
        else:
            st.error("❌ Free uses exhausted. Add an API key or use secret API in sidebar.")
    
    api_key_input = st.text_input(
        "Google AI API Key (session-only)",
        type="password",
        value=st.session_state.get("custom_api_key", ""),
        help="Leave empty to use secret API. Changes only affect this session.",
        placeholder="Leave empty to use secret API..."
    )
    
    # Update API key for this session if input changed
    if api_key_input != st.session_state.get("custom_api_key", ""):
        st.session_state["custom_api_key"] = api_key_input
        
        if api_key_input and api_key_input.strip():
            # User provided a custom API key
            utility.set_api_key(api_key_input.strip())
            st.success("✅ API key updated for this session!")
        else:
            # User cleared the field - use secret API (environment variable)
            utility.set_api_key(None)  # This makes get_api_key() fall back to env var
            st.info("ℹ️ Using secret API from environment")
        
        st.rerun()
    
    # Ensure API key is set on each render
    if custom_api_key:
        utility.set_api_key(custom_api_key)
    else:
        utility.set_api_key(None)
    
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

    st.divider()

    st.subheader("📎 PDF Context")
    if st.session_state.get("uploaded_pdf"):
        st.success(f"Attached: {st.session_state['uploaded_pdf']}")
        if st.button("✕ Remove PDF", key="remove_pdf_sidebar", use_container_width=True):
            st.session_state["uploaded_pdf"] = None
            st.session_state["pdf_text"] = None
            st.session_state["pdf_path"] = None
            st.session_state["had_pdf_context"] = False
            st.session_state.pop("pdf_uploader", None)
            st.rerun()

    st.caption("Attach an optional PDF to give the assistant extra context.")
    uploaded_file = st.file_uploader(
        "Upload PDF for context",
        type=["pdf"],
        help="Attach a PDF to ground responses.",
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None and st.session_state.get("uploaded_pdf") != uploaded_file.name:
        try:
            temp_pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_text = extract_text_from_pdf(temp_pdf_path)

            if pdf_text.startswith("ERROR:"):
                st.error(f"Failed to read PDF: {pdf_text}")
                st.session_state["uploaded_pdf"] = None
                st.session_state["pdf_text"] = None
            else:
                st.session_state["uploaded_pdf"] = uploaded_file.name
                st.session_state["pdf_text"] = pdf_text
                st.session_state["pdf_path"] = temp_pdf_path
                st.session_state["had_pdf_context"] = True
                st.rerun()
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state["uploaded_pdf"] = None
            st.session_state["pdf_text"] = None

error = st.session_state.get("last_error")
if error:
    st.error(error)

if st.session_state.get("uploaded_pdf"):
    st.markdown(
        f"""
        <div class='pdf-attachment-card'>
            <div class='pdf-icon'>📄</div>
            <div class='pdf-details'>
                <div style='color: white; font-weight: 600; font-size: 14px;'>{st.session_state['uploaded_pdf']}</div>
                <div style='color: #bbb; font-size: 12px;'>Attached PDF context</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for conv in st.session_state.conversations:
    _render_conversation(conv)

# If currently processing a prompt, show it first
if st.session_state.get("current_prompt"):
    with st.chat_message("user"):
        display_prompt = st.session_state.get("current_prompt_display", st.session_state["current_prompt"])
        st.markdown(display_prompt)

prompt_input = st.chat_input("Ask anything")

if prompt_input:
    # Check usage limits (only for free tier - no custom API key in this session)
    usage_data = st.session_state["usage_data"]
    has_api_key = st.session_state.get("custom_api_key", "").strip()
    
    if not has_api_key and usage_data.get("uses", 0) >= 3:
        st.error("⚠️ You've used all 3 free queries. Add an API key in sidebar or use secret API.")
        st.stop()
    
    # Build intelligent conversation context by extracting relevant info from history
    context_summary = _build_conversation_context(prompt_input, st.session_state.get("conversations", []))
    
    # Add PDF content if uploaded
    pdf_context = ""
    if st.session_state.get("pdf_text"):
        pdf_context = f"""
[Attached PDF: {st.session_state['uploaded_pdf']}]
---
{st.session_state['pdf_text']}
---

"""
    
    # Compose final prompt with context + PDF + query
    if context_summary:
        composed = f"[Conversation Context]\n{context_summary}\n\n{pdf_context}[Current Query]\n{prompt_input}"
    elif pdf_context:
        composed = f"{pdf_context}[Current Query]\n{prompt_input}"
    else:
        composed = prompt_input

    # Store BOTH the original prompt (for display) and the composed prompt (for processing)
    st.session_state["current_prompt_display"] = prompt_input  # What user sees
    st.session_state["current_prompt"] = composed  # What pipeline processes
    st.session_state["had_pdf_context"] = bool(pdf_context)  # Track if this message had PDF context
    st.session_state["processing_complete"] = False
    st.rerun()

# Process the stored prompt if it exists and we haven't processed it yet
if st.session_state.get("current_prompt") and not st.session_state.get("processing_complete"):
    with st.spinner("Running pipeline..."):
        try:
            result = _run_pipeline(st.session_state["current_prompt"], st.session_state["verbose"])
            
            # Increment usage count only on success (if no custom API key this session)
            if not st.session_state.get("custom_api_key", "").strip():
                st.session_state["usage_data"]["uses"] = st.session_state["usage_data"].get("uses", 0) + 1
                
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
            
            # Auto-create PDF ONLY if this message had PDF context
            had_pdf_context = st.session_state.get("had_pdf_context", False)
            if had_pdf_context:
                original_prompt = st.session_state.get("current_prompt_display", st.session_state["current_prompt"])
                if _should_create_pdf(original_prompt, result):
                    pdf_path = _create_pdf_from_result(original_prompt, result)
                    if pdf_path:
                        conversation["pdf_path"] = pdf_path
            
            st.session_state.conversations.append(conversation)
            st.session_state.selected_blocks[conversation["id"]] = None
            st.session_state.last_error = None
            st.session_state["current_prompt"] = None
            st.session_state["current_prompt_display"] = None  # Clear display version too
            st.session_state["had_pdf_context"] = False  # Reset PDF context flag
            st.session_state["processing_complete"] = True
            
            # Clear uploaded PDF after processing (optional - can be removed if you want it to persist)
            # st.session_state["uploaded_pdf"] = None
            # st.session_state["pdf_text"] = None
            # st.session_state["pdf_path"] = None
            
            st.rerun()
