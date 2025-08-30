import os
import time
import json
import streamlit as st
from chain_of_thought import chain_of_thought, plan_prompt, critique_plan_prompt, fixer_plan_prompt, executer_prompt, scorer_prompt, improver_prompt, digest_prompt, init_model

st.set_page_config(page_title="Thinking Without Thinking", page_icon="🧠", layout="wide")

if "calls_used" not in st.session_state:
    st.session_state.calls_used = 0
if "use_custom_key" not in st.session_state:
    st.session_state.use_custom_key = False

try:
    primary_key = st.secrets.get("GOOGLE_API_KEY", None)
except:
    primary_key = None

def configure_api_key():
    key = primary_key
    if st.session_state.use_custom_key:
        key = st.session_state.get("custom_key")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return True
    return False

def call_guard() -> bool:
    if st.session_state.use_custom_key:
        return True
    if primary_key is None:
        return True
    if st.session_state.calls_used < 3:
        st.session_state.calls_used += 1
        return True
    return False

def parse_json_score(score_raw):
    try:
        score_data = json.loads(score_raw)
        clarity = score_data.get("clarity", -1)
        logic = score_data.get("logic", -1)
        actionability = score_data.get("actionability", -1)
        feedback = score_data.get("feedback", "No feedback provided")
        composite_score = (clarity * 2 + logic * 4 + actionability * 2) / 8 * 10
        return clarity, logic, actionability, feedback, composite_score
    except (json.JSONDecodeError, KeyError):
        try:
            parts = score_raw.split(" | ")
            feedback, composite_score = parts[0], int(parts[1])
            return -1, -1, -1, feedback, composite_score
        except (ValueError, IndexError):
            return -1, -1, -1, score_raw, -1

accent = "#5b9bd5"
crit = "#ff7f50"
fix = "#22c55e"
resp = "#facc15"
score = "#e879f9"
muted = "#64748b"

st.markdown(f"""
    <style>
    .section-title {{
        font-weight: 700;
        color: {accent};
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        letter-spacing: .3px;
    }}
    .pill {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        background: {accent};
        margin-right: .5rem;
    }}
    .box {{
        background: #0b0f19;
        border: 1px solid #1f2937;
        border-radius: .6rem;
        padding: .9rem;
        margin-bottom: .8rem;
    }}
    .crit {{ border-color: {crit}; }}
    .fix {{ border-color: {fix}; }}
    .resp {{ border-color: {resp}; }}
    .score {{ border-color: {score}; }}
    .muted {{ color: {muted}; }}
    .score-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0;
    }}
    .score-item {{
        background: #1a1a2e;
        padding: 0.5rem;
        border-radius: 0.375rem;
        text-align: center;
        border: 1px solid #374151;
    }}
    .score-value {{
        font-size: 1.25rem;
        font-weight: bold;
        color: {accent};
    }}
    .score-label {{
        font-size: 0.75rem;
        color: {muted};
        text-transform: uppercase;
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Reasoning App")
    st.caption("Each user can use the built-in key up to 3 times. Add your own key for unlimited use.")
    st.toggle("Use my own Google API key", key="use_custom_key")
    if st.session_state.use_custom_key:
        st.text_input("Enter your Google API Key", key="custom_key", type="password")
        if st.button("Save Key"):
            if configure_api_key():
                st.success("API key set")
            else:
                st.error("Invalid key")
    st.divider()
    st.subheader("Get a Google API Key")
    st.markdown("1. Go to https://aistudio.google.com/app/apikey")
    st.markdown("2. Sign in and create an API key")
    st.markdown("3. Copy it here when prompted")
    st.divider()
    st.markdown(f"<span class='pill'>Free key uses left: {max(0, 3-st.session_state.calls_used)}</span>", unsafe_allow_html=True)

st.title("Thinking Without Thinking")
st.write("Enter a prompt and view the full reasoning trace.")

with st.form("input_form"):
    prompt = st.text_area("Your prompt", height=180, placeholder="Paste text or ask a question…")
    show_trace = st.checkbox("Show chain of thought sections", value=True)
    submitted = st.form_submit_button("Run Reasoning")

if submitted:
    if not configure_api_key():
        st.error("No API key configured. Add one in the sidebar.")
    elif not call_guard():
        st.error("Free quota for the built-in key is used up. Add your own key in the sidebar.")
    elif not prompt.strip():
        st.error("Enter a prompt")
    else:
        ph = st.empty()
        with st.spinner("Thinking…"):
            try:
                use_digest = len(prompt) > int(os.getenv("COT_DIGEST_THRESHOLD_CHARS", "1500"))
                ctx = digest_prompt(prompt, verbose=False) if use_digest else prompt
                plan = plan_prompt(ctx, verbose=False)
                crit = critique_plan_prompt(ctx, plan, verbose=False)
                fixed = fixer_plan_prompt(ctx, plan, crit, verbose=False)
                resp = executer_prompt(ctx, fixed, verbose=False)
                score_raw = scorer_prompt(ctx, resp, verbose=False)
                
                clarity, logic, actionability, feedback, composite_score = parse_json_score(score_raw)
                
                improved = None
                if composite_score < 80 or (logic != -1 and logic < 6):
                    improved = improver_prompt(ctx, resp, fixed, score_raw, verbose=False)
                    
            except Exception as e:
                error_str = str(e)
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
                    import re
                    delay_match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
                    if delay_match:
                        retry_seconds = int(delay_match.group(1))
                        st.info(f"⏱️ Suggested retry delay: {retry_seconds} seconds")
                else:
                    st.error(f"❌ **Unexpected Error**: {error_str}")
                raise
                
        st.success("Done")
        with st.container():
            if use_digest:
                st.markdown("<div class='section-title'>DIGEST</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='box'>{ctx}</div>", unsafe_allow_html=True)
            
            if show_trace:
                st.markdown("<div class='section-title'>PLAN</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='box'>{plan}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>CRITIQUE</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='box crit'>{crit}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>FIXED PLAN</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='box fix'>{fixed}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-title'>RESPONSE</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='box resp'>{resp}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-title'>EVALUATION</div>", unsafe_allow_html=True)
            if clarity != -1:
                score_html = f"""
                <div class='box score'>
                    <div class='score-grid'>
                        <div class='score-item'>
                            <div class='score-value'>{clarity}/10</div>
                            <div class='score-label'>Clarity</div>
                        </div>
                        <div class='score-item'>
                            <div class='score-value'>{logic}/10</div>
                            <div class='score-label'>Logic</div>
                        </div>
                        <div class='score-item'>
                            <div class='score-value'>{actionability}/10</div>
                            <div class='score-label'>Actionability</div>
                        </div>
                    </div>
                    <p><strong>Composite Score:</strong> {composite_score:.1f}/100</p>
                    <p><strong>Feedback:</strong> {feedback}</p>
                </div>
                """
                st.markdown(score_html, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='box score'>{feedback}<br><br><strong>Score:</strong> {composite_score:.1f}/100</div>", unsafe_allow_html=True)
            
            if improved:
                st.markdown("<div class='section-title'>IMPROVED RESPONSE</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='box resp'>{improved}</div>", unsafe_allow_html=True)
