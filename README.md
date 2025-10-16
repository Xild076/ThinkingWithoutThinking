# Thinking Without Thinking

This is just a fun little project to make a streamlit app that somewhat simulates "thought" and "chain of reasoning" with prompt engineering. Its aim is to mirror some aspect of human psychology with how we come up with structured responses to problems. It currently uses the Gemma 27B model, so if you get a google API key, its free to use.

I took heavy inspiration from an old GPT prompt engineering widget where it asked the LLM to critique itself. However, instead of single-prompt analysis which has the caveat of severe biases, I implemented a cross-prompt analysis. While its more computationally expensive, it's also a much more accurate version of the prompt engineering.

## What it Does

ThinkingWithoutThinking implements a sophisticated multi-step reasoning system that breaks down complex problems into structured thinking processes. The system uses a systematic thinking approach with multiple thinking "blocks" to synthesize a final, high quality response to user queries.

## Core Features

1. **Agentic Reasoning Pipeline:** Decomposes complex prompts into a logical sequence of steps—planning, web research, code execution, and synthesis—managed by a central orchestrator.
2. **Executable Code Tool:** A sandboxed Python environment that can write and run code to perform data analysis, mathematical calculations, and generate visualizations.
3. **Live Web Search:** Implements a web search and content extraction tool to answer questions with relevant information, grounding responses.
4. **Self-Correction:** Features feedback loops at every critical stage: initial outputs are critiqued and then refined by a built-in process.
5. **Transparency:** The Streamlit UI provides a complete, interactive trace of the agent's reasoning process. Inspect every tool's input/output, view generated code, and understand exactly how the final answer was produced.

## How it works

The pipeline follows a structured, four-phase process for every query:
- **Phase 1:** Plan & Route (Meta-Reasoning)
    - **Plan Creation:** The system first generates a high-level, human-readable strategy to address the user's prompt. This plan is then critiqued and refined in a self-correction step.
    - **Routing:** The refined plan is translated into a machine-readable sequence of "blocks" (e.g., use_internet_tool -> use_code_tool -> synthesize_final_answer), creating a custom pipeline tailored to the specific query.
- **Phase 2:** Execute (Tool Use)
    - The pipeline executes each block in sequence.
        - The Internet Tool scrapes web pages for relevant information.
        - The Code Tool writes and runs Python code to analyze data or create plots.
        - The Creative Idea Generation Tool uses high temperature to create creative concepts 
    - The output of each block is stored in a shared context, allowing subsequent blocks to build on previous results.
**Phase 3:** Synthesize & Refine (Answer Generation)
    - The SynthesizeFinalAnswerBlock gathers all the information from the context from tool use.
    - The block generates a comprehensive, user-facing answer in Markdown.
    - This initial answer is then passed to a Scorer prompt, which rates it on clarity, logic, and actionability.
    - Finally, an Improver prompt refines the response based on the scorer's feedback, producing the final, polished result.
**Phase 4:** Present (User Interface)
The entire process is visualized in an interactive Streamlit dashboard. Users can see live progress, inspect each step, and view the final answer with any generated artifacts like charts.

## Usage

### Prerequisites
- Python 3.12
- A Google Generative AI API Key (https://aistudio.google.com/api-keys, free tier works)

### Installation
```bash
git clone https://github.com/your-username/ThinkingWithoutThinking.git
cd ThinkingWithoutThinking

python -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure API Key

1. Use the Streamlit UI (recommended): You can enter your API key in the sidebar to use.
2. Set an environmental variable:
```bash
# On macOS/Linux
export GOOGLE_API_KEY="your-api-key"

# On Windows (PowerShell)
$Env:GOOGLE_API_KEY = "your-api-key"
```

### Usage
1. Use the live app: https://thinking-gemma.streamlit.app/
2. Use the Streamlit UI: 
```bash 
streamlit run app.py
```
3. Use the CLI:
```bash
python -m src.main "GRAPH the function y = x^3 - 2x^2 + 5 for x from -10 to 10"
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue to discuss your ideas or submit a pull request.
