"""
Streamlit UI for ThinkingWithoutThinking Pipeline.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ThinkingWithoutThinking",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a0b;
    }
    
    .main-header {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #71717a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .step-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .step-card.active {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    
    .step-icon {
        font-size: 1.5rem;
    }
    
    .step-title {
        font-weight: 600;
        color: #fafafa;
    }
    
    .step-content {
        background-color: #0a0a0b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #a1a1aa;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .final-response {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid #8b5cf6;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
    
    .stat-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fafafa;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #71717a;
        text-transform: uppercase;
    }
    
    .error-box {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        color: #fafafa;
    }
    
    .warning-box {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_run' not in st.session_state:
    st.session_state.current_run = None
if 'cost_stats' not in st.session_state:
    st.session_state.cost_stats = None

# API base URL
API_BASE = "http://localhost:8000"

# Step icons
STEP_ICONS = {
    'planning': '📝',
    'plan': '📋',
    'critique': '🔍',
    'plan_critique': '🔍',
    'improvement': '✨',
    'plan_improved': '✨',
    'routing': '🔀',
    'routed': '🔀',
    'tool_start': '⚙️',
    'tool_complete': '✅',
    'tool_error': '❌',
    'synthesizing': '🔮',
    'response': '💬',
    'final_critique': '🔍',
    'final_improvement': '✨',
    'final_response': '🎯',
    'complete': '✅',
    'error': '❌',
    'warning': '⚠️'
}

STEP_TITLES = {
    'planning': 'Generating Plan',
    'plan': 'Plan Generated',
    'critique': 'Self Critique',
    'plan_critique': 'Plan Critique',
    'improvement': 'Improving Plan',
    'plan_improved': 'Plan Improved',
    'routing': 'Routing to Tools',
    'routed': 'Tools Selected',
    'tool_start': 'Tool Starting',
    'tool_complete': 'Tool Complete',
    'tool_error': 'Tool Failed',
    'synthesizing': 'Synthesizing Response',
    'response': 'Initial Response',
    'final_critique': 'Final Critique',
    'final_improvement': 'Final Improvement',
    'final_response': 'Final Response',
    'complete': 'Complete',
    'error': 'Error',
    'warning': 'Warning'
}


def fetch_cost_stats():
    """Fetch cost statistics from the API."""
    try:
        response = requests.get(f"{API_BASE}/cost-stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def reload_prompts():
    """Reload prompts from disk via API."""
    try:
        response = requests.post(f"{API_BASE}/reload-prompts", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Unknown error"}


def stream_pipeline(prompt: str, thinking_level: str):
    """Stream pipeline responses from the API."""
    try:
        response = requests.post(
            f"{API_BASE}/stream",
            json={"prompt": prompt, "thinking_level": thinking_level},
            stream=True,
            timeout=120
        )
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"type": "error", "message": str(e)}


def render_step(event: dict, is_active: bool = False):
    """Render a pipeline step."""
    event_type = event.get('type', 'unknown')
    icon = STEP_ICONS.get(event_type, '📌')
    title = STEP_TITLES.get(event_type, event_type.title())
    
    # Determine content
    content = None
    if 'plan' in event:
        content = event['plan']
    elif 'critique' in event:
        content = event['critique']
    elif 'response' in event:
        content = event['response']
    elif 'message' in event and event_type not in ['planning', 'critique', 'improvement', 'routing', 'synthesizing']:
        content = event['message']
    elif 'tools' in event:
        content = json.dumps(event['tools'], indent=2)
    elif 'result' in event:
        result = event['result']
        content = result if isinstance(result, str) else json.dumps(result, indent=2)
    elif 'error' in event:
        content = event['error']
    
    # Build subtitle
    subtitle = ""
    if 'tool_id' in event:
        subtitle = event['tool_id']
    elif 'message' in event and event_type in ['planning', 'critique', 'improvement', 'routing', 'synthesizing']:
        subtitle = event['message']
    
    # Render
    card_class = "step-card active" if is_active else "step-card"
    if event_type == 'error':
        card_class += " error"
    
    with st.container():
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown(f"<span style='font-size: 1.5rem;'>{icon}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}**" + (f" - {subtitle}" if subtitle else ""))
            if content:
                with st.expander("View Details", expanded=event_type in ['final_response', 'error']):
                    if len(content) > 2000:
                        st.code(content[:2000] + "\n\n... (truncated)", language=None)
                    else:
                        st.code(content, language=None)


def render_final_response(response: str):
    """Render the final response prominently."""
    st.markdown("---")
    st.markdown("### 🎯 Final Response")
    st.markdown(
        f"""<div class="final-response">
        <p style="color: #fafafa; font-size: 1rem; line-height: 1.7;">{response}</p>
        </div>""",
        unsafe_allow_html=True
    )
    # Also show as regular text for better readability
    st.write(response)


# Sidebar
with st.sidebar:
    st.markdown('<h1 class="main-header">🧠 ThinkingWithoutThinking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI Reasoning Pipeline</p>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("### 💬 Your Prompt")
    prompt = st.text_area(
        label="prompt",
        label_visibility="collapsed",
        placeholder="Ask me anything... I'll think through it step by step.",
        height=150
    )
    
    thinking_level = st.selectbox(
        "Thinking Level",
        options=["low", "medium_synth", "medium_plan", "high"],
        index=1,
        format_func=lambda x: {
            "low": "🚀 Low - Quick response",
            "medium_synth": "⚡ Medium - With synthesis",
            "medium_plan": "📋 Medium - With planning",
            "high": "🧠 High - Full reasoning"
        }[x]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button("🚀 Generate", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.current_run = None
        st.rerun()
    
    st.markdown("---")
    
    # Stats section
    st.markdown("### 📊 Cost Tracking")
    stats = fetch_cost_stats()
    if stats and stats.get('status') == 'ok':
        st.session_state.cost_stats = stats
        session = stats['session']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tokens In", f"{session['total_input_tokens']:,}")
        with col2:
            st.metric("Tokens Out", f"{session['total_output_tokens']:,}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Est. Cost", f"${session['estimated_cost']:.4f}")
        with col2:
            st.metric("Avg Latency", f"{session['average_latency_ms']:.0f}ms")
        
        st.metric("Total Calls", session['total_calls'])
    else:
        st.info("Connect to backend to see stats")
    
    st.markdown("---")
    
    # Developer tools
    st.markdown("### 🔧 Developer Tools")
    if st.button("🔄 Reload Prompts", use_container_width=True):
        result = reload_prompts()
        if result.get('status') == 'ok':
            st.success(f"Reloaded {result.get('prompt_count', 0)} prompts")
        else:
            st.error(result.get('message', 'Unknown error'))
    
    if st.button("📊 Refresh Stats", use_container_width=True):
        st.rerun()


# Main content area
st.markdown('<h1 class="main-header">AI Reasoning Pipeline</h1>', unsafe_allow_html=True)

if submit_button and prompt:
    # Create a placeholder for streaming output
    output_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = []
    final_response = None
    tool_errors = []
    
    with output_container:
        step_placeholder = st.empty()
        
        for i, event in enumerate(stream_pipeline(prompt, thinking_level)):
            event_type = event.get('type', 'unknown')
            
            # Update progress
            progress = min(0.1 + (i * 0.1), 0.9)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {STEP_TITLES.get(event_type, event_type)}")
            
            # Collect events
            steps.append(event)
            
            # Handle special events
            if event_type == 'final_response':
                final_response = event.get('response')
            elif event_type == 'complete':
                if event.get('final', {}).get('response'):
                    final_response = event['final']['response']
                if event.get('final', {}).get('tool_errors'):
                    tool_errors = event['final']['tool_errors']
            elif event_type == 'tool_error':
                tool_errors.append(event)
            
            # Re-render all steps
            with step_placeholder.container():
                for step_event in steps[-5:]:  # Show last 5 steps while streaming
                    render_step(step_event, is_active=(step_event == steps[-1]))
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
    
    # Clear streaming display and show final results
    step_placeholder.empty()
    progress_bar.empty()
    status_text.empty()
    
    # Show tool errors if any
    if tool_errors:
        st.markdown("### ⚠️ Tool Errors")
        for err in tool_errors:
            st.error(f"**{err.get('tool_id', 'Unknown')}**: {err.get('error', 'Unknown error')}")
    
    # Show all steps in expanders
    st.markdown("### 📋 Pipeline Steps")
    for step in steps:
        if step.get('type') not in ['complete']:
            render_step(step)
    
    # Show final response prominently
    if final_response:
        render_final_response(final_response)
    
    # Store in history
    st.session_state.history.append({
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'thinking_level': thinking_level,
        'steps': steps,
        'final_response': final_response
    })

elif st.session_state.current_run:
    # Display previous run
    run = st.session_state.current_run
    st.markdown(f"**Previous prompt:** {run['prompt']}")
    for step in run.get('steps', []):
        render_step(step)
    if run.get('final_response'):
        render_final_response(run['final_response'])

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px; color: #71717a;">
        <p style="font-size: 4rem; margin-bottom: 16px;">💭</p>
        <h2 style="color: #a1a1aa; margin-bottom: 8px;">Ready to Think</h2>
        <p>Enter a prompt in the sidebar and I'll walk you through my reasoning process step by step.</p>
    </div>
    """, unsafe_allow_html=True)


# History section (collapsible)
if st.session_state.history:
    with st.expander("📜 Query History"):
        for i, run in enumerate(reversed(st.session_state.history[-10:])):  # Last 10 runs
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{run['timestamp'][:19]}** - {run['prompt'][:50]}...")
            with col2:
                if st.button("View", key=f"view_{i}"):
                    st.session_state.current_run = run
                    st.rerun()
