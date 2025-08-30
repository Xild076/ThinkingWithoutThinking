import os
import time
import streamlit as st
from src.chain_of_thought import plan_prompt, critique_plan_prompt, fixer_plan_prompt, executer_prompt, scorer_prompt, improver_prompt, digest_prompt, init_model

st.set_page_config(page_title="Thinking Without Thinking", page_icon="🧠", layout="centered")

if "calls_used" not in st.session_state:
    st.session_state.calls_used = 0
if "conversation" not in st.session_state:
    st.session_state.conversation = []

primary_key = st.secrets.get("GOOGLE_API_KEY", None)

def configure_api_key():
    key = primary_key
    if st.session_state.get("custom_key"):
        key = st.session_state.custom_key
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        init_model()
        return True
    return False

def call_guard() -> bool:
    if st.session_state.get("custom_key"):
        return True
    if primary_key is None:
        st.error("No shared API key available.")
        return False
    if st.session_state.calls_used < 3:
        st.session_state.calls_used += 1
        return True
    return False

def build_context_prompt(current_prompt):
    if len(st.session_state.conversation) <= 1:
        return current_prompt
    
    context_parts = []
    for msg in st.session_state.conversation[:-1]:
        if 'prompt' in msg and 'final_response' in msg:
            context_parts.append(f"Previous Q: {msg['prompt'][:200]}...")
            context_parts.append(f"Previous A: {msg['final_response'][:200]}...")
    
    if context_parts:
        context = "\n".join(context_parts[-4:])
        if len(context) > 1500:
            try:
                context = digest_prompt(f"Summarize this conversation context concisely:\n{context}", verbose=False)
            except:
                context = context[:1500] + "..."
        return f"Previous conversation context:\n{context}\n\nCurrent request: {current_prompt}"
    return current_prompt

st.markdown("""
    <style>
    /* Theme-aware styling that adapts to light/dark mode */
    .user-message {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 16px;
        border-radius: 18px;
        margin: 8px 0 8px 20%;
        border: 1px solid var(--border-color);
    }
    .bot-message {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 16px;
        border-radius: 18px;
        margin: 8px 20% 8px 0;
        border: 1px solid var(--border-color);
    }
    .thinking-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    .reasoning-item {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin: 6px 0;
        overflow: hidden;
    }
    .reasoning-header {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 10px 14px;
        font-weight: 600;
        border-bottom: 1px solid var(--border-color);
    }
    .reasoning-content {
        padding: 14px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 13px;
        line-height: 1.4;
        white-space: pre-wrap;
        color: var(--text-color);
        background-color: var(--secondary-background-color);
    }
    
    /* Light mode colors */
    [data-theme="light"] {
        --background-color: #f7f7f8;
        --secondary-background-color: white;
        --text-color: #262626;
        --border-color: #e5e5e7;
    }
    
    /* Dark mode colors */
    [data-theme="dark"] {
        --background-color: #262730;
        --secondary-background-color: #1e1e1e;
        --text-color: #ffffff;
        --border-color: #404040;
    }
    
    /* Fallback for Streamlit's automatic theme detection */
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #f7f7f8;
            --secondary-background-color: white;
            --text-color: #262626;
            --border-color: #e5e5e7;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #262730;
            --secondary-background-color: #1e1e1e;
            --text-color: #ffffff;
            --border-color: #404040;
        }
    }
    
    /* Ensure text color inheritance */
    .user-message *, .bot-message *, .thinking-box *, .reasoning-item *, .reasoning-content * {
        color: inherit !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🧠 Neural Reasoning")
    
    use_custom = st.toggle("Use my own Google API key")
    
    if use_custom:
        st.text_input("🔑 API Key", key="custom_key", type="password", placeholder="Enter your key...")
        if st.button("💾 Save Key", use_container_width=True):
            if configure_api_key():
                st.success("✅ Key configured")
            else:
                st.error("❌ Invalid key")
    
    with st.expander("🔗 Get Google AI API Key"):
        st.write("1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)")
        st.write("2. Sign in and create a new API key")
        st.write("3. Copy and paste it here")
    
    st.divider()
    
    # Rate limiting configuration
    with st.expander("⚙️ Advanced Settings"):
        st.slider("Rate limit safety delay (seconds)", 
                 min_value=0, max_value=120, value=30, step=5,
                 key="rate_limit_delay",
                 help="Add delay between requests to avoid rate limits")
        st.toggle("Auto-retry on rate limits", value=True, key="auto_retry",
                 help="Automatically retry failed requests due to rate limits")
    
    st.divider()
    remaining = max(0, 3 - st.session_state.calls_used)
    st.info(f"Free uses remaining: {remaining}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

st.title("Thinking Without Thinking")
st.caption("Advanced chain-of-thought reasoning with transparency")

for i, msg in enumerate(st.session_state.conversation):
    if 'prompt' in msg:
        st.markdown(f"<div class='user-message'><strong>You</strong><br>{msg['prompt']}</div>", unsafe_allow_html=True)
    
    if 'final_response' in msg:
        st.markdown(f"<div class='bot-message'><strong>Reasoning Gemma</strong><br>{msg['final_response']}</div>", unsafe_allow_html=True)
        if msg.get('plan'):
            with st.expander("🔍 View reasoning steps"):
                steps = [
                    ("digest", "📝 Input Analysis"),
                    ("plan", "📋 Planning"), 
                    ("critique", "⚠️ Critique"),
                    ("fixed", "🔧 Refinement"),
                    ("response", "💡 Response Generation"),
                    ("score", "📊 Quality Assessment"),
                    ("improved", "✨ Enhancement")
                ]
                for step_key, step_name in steps:
                    if step_key in msg and msg[step_key]:
                        st.markdown(f"""
                            <div class='reasoning-item'>
                                <div class='reasoning-header'>{step_name}</div>
                                <div class='reasoning-content'>{msg[step_key]}</div>
                            </div>
                        """, unsafe_allow_html=True)

if prompt := st.chat_input("Message Reasoning Gemma..."):
    if not configure_api_key():
        st.error("🔑 Please configure an API key in the sidebar")
    elif not call_guard():
        st.error("🚫 Free quota exhausted. Add your own API key to continue")
    else:
        st.session_state.conversation.append({"prompt": prompt})
        st.rerun()

if st.session_state.conversation and 'final_response' not in st.session_state.conversation[-1]:
    msg = st.session_state.conversation[-1]
    prompt_text = msg['prompt']
    
    st.markdown(f"<div class='user-message'><strong>You</strong><br>{prompt_text}</div>", unsafe_allow_html=True)
    
    progress_container = st.empty()
    
    steps = [
        ("📝", "Analyzing input"),
        ("📋", "Creating plan"), 
        ("⚠️", "Finding issues"),
        ("🔧", "Refining approach"),
        ("💡", "Generating response"),
        ("📊", "Assessing quality"),
        ("✨", "Enhancing output")
    ]
    
    current_step = 0

    def show_progress(step_idx, status="active"):
        with progress_container.container():
            st.markdown("<div class='thinking-box'><strong>🤔 Thinking...</strong></div>", unsafe_allow_html=True)
            
            for i, (icon, desc) in enumerate(steps):
                if i < step_idx:
                    st.markdown(f"✅ {desc}", unsafe_allow_html=True)
                elif i == step_idx and status == "active":
                    st.markdown(f"🔄 {desc} (in progress...)", unsafe_allow_html=True)
                elif i == step_idx and status == "complete":
                    st.markdown(f"✅ {desc}", unsafe_allow_html=True)
                else:
                    st.markdown(f"⏳ {desc}", unsafe_allow_html=True)

    try:
        context_prompt = build_context_prompt(prompt_text)
        
        # Apply user-configured rate limiting
        if st.session_state.get("rate_limit_delay", 30) > 0:
            time.sleep(st.session_state.rate_limit_delay / 6)  # Distribute delay across steps
        
        current_step = 1
        show_progress(current_step, "active")
        plan = plan_prompt(context_prompt, verbose=False)
        msg["plan"] = plan
        show_progress(current_step, "complete")
        
        current_step = 2
        show_progress(current_step, "active")
        crit = critique_plan_prompt(context_prompt, plan, verbose=False)
        msg["critique"] = crit
        show_progress(current_step, "complete")
        
        current_step = 3
        show_progress(current_step, "active")
        fixed = fixer_plan_prompt(context_prompt, plan, crit, verbose=False)
        msg["fixed"] = fixed
        show_progress(current_step, "complete")
        
        current_step = 4
        show_progress(current_step, "active")
        resp = executer_prompt(context_prompt, fixed, verbose=False)
        msg["response"] = resp
        show_progress(current_step, "complete")
        
        current_step = 5
        show_progress(current_step, "active")
        sc = scorer_prompt(context_prompt, resp, verbose=False)
        try:
            parts = sc.split(" | ")
            feedback, score_int = parts[0], int(parts[1])
        except (ValueError, IndexError):
            feedback, score_int = sc, -1
        msg["score"] = f"{feedback}\n\nScore: {score_int}"
        show_progress(current_step, "complete")
        
        final_response = resp
        if score_int < 80:
            current_step = 6
            show_progress(current_step, "active")
            improved = improver_prompt(context_prompt, resp, fixed, feedback, verbose=False)
            msg["improved"] = improved
            final_response = improved
            show_progress(current_step, "complete")
        
        msg["final_response"] = final_response
        
    except Exception as e:
        error_str = str(e)
        
        # Check for rate limit errors
        if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
            st.error("🚫 **Rate Limit Exceeded**")
            st.warning("""
            The Google API rate limit has been exceeded. This can happen when:
            - Too many requests are made in a short time
            - Token quota is exhausted
            
            **What to do:**
            - Wait a minute and try again
            - Try using shorter prompts to reduce token usage
            - Consider upgrading your Google API plan for higher limits
            """)
            
            # Extract retry delay if available
            import re
            delay_match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
            if delay_match:
                retry_seconds = int(delay_match.group(1))
                st.info(f"⏱️ Suggested retry delay: {retry_seconds} seconds")
            
            msg["final_response"] = "Request failed due to rate limit. Please wait and try again with a shorter prompt."
        else:
            st.error(f"❌ **Unexpected Error**: {error_str}")
            msg["final_response"] = f"An error occurred while processing your request: {error_str}"
    
    progress_container.empty()
    st.rerun()
