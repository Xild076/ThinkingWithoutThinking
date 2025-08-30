# ThinkingWithoutThinking

This is just a fun little project to make a streamlit app that someone simulates "thought" and "chain of reasoning" with prompt engineering. It currently uses the Gemma 27B model, so if you get a google API key, its free to use. Have fun!

## What It Does

ThinkingWithoutThinking implements a sophisticated multi-step reasoning system that breaks down complex problems into structured thinking processes. The system uses a chain-of-thought approach with multiple AI agents that plan, critique, fix, execute, score, and improve responses to ensure high-quality, logically consistent outputs.

## Features

### **Multi-Step Reasoning Chain**
- **Plan**: Creates a structured execution plan
- **Critique**: Reviews the plan for potential issues
- **Fix**: Refines the plan based on critique
- **Execute**: Implements the plan to generate a response
- **Score**: Evaluates the response across multiple dimensions
- **Improve**: Enhances the response if needed

### **Advanced Scoring System**
- **Multi-dimensional evaluation**: Clarity, Logic, Actionability
- **Logic-weighted scoring**: Prioritizes logical consistency
- **Smart improvement triggers**: Automatic enhancement for low logic scores
- **Visual score breakdown**: Clear display of individual dimension scores

### **Robust Error Handling**
- **Rate limit protection**: Automatic retry with intelligent backoff
- **Quota management**: Proactive token usage estimation
- **User-friendly errors**: Clear explanations instead of technical jargon
- **Graceful degradation**: Continues working even with API issues

### **Clean User Interface**
- **Streamlit interface**: Modern, responsive web interface
- **Interactive trace view**: Optional display of reasoning steps
- **Real-time feedback**: Progress indicators and status updates
- **Dark theme**: Easy on the eyes for extended use

## How to use

### Prerequisites
- Python 3.8+
- Google AI API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ThinkingWithoutThinking.git
   cd ThinkingWithoutThinking
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get your Google AI API key**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in and create a new API key
   - Copy the key for the next step

5. **Configure your API key**
   
   **Option 1: Environment Variable**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   
   **Option 2: Streamlit Secrets** (Recommended for deployment)
   ```bash
   mkdir .streamlit
   echo 'GOOGLE_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```
   
   **Option 3: In-App Configuration**
   - Use the sidebar in the app to enter your key

### Running the App

**Streamlit Interface:**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python -m src.chain_of_thought "Your question here"
```

## 📁 Project Structure

```
ThinkingWithoutThinking/
├── src/
│   ├── __init__.py
│   ├── app.py                 # Alternative Streamlit interface
│   └── chain_of_thought.py    # Core reasoning engine
├── app.py                     # Main Streamlit application
├── test_logical_consistency.py # Test suite for logical consistency
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key
- `GENAI_MODEL`: Model to use (default: "gemma-3-27b-it")
- `GENAI_MAX_OUTPUT_TOKENS`: Maximum tokens per request (default: 8192)
- `COT_DIGEST_THRESHOLD_CHARS`: When to summarize long inputs (default: 1500)

### App Settings
- **Rate limit safety delay**: 0-120 seconds between requests
- **Auto-retry on rate limits**: Enable/disable automatic retries
- **Show chain of thought**: Toggle detailed reasoning trace

## 🧪 Testing

Run the logical consistency test:
```bash
python test_logical_consistency.py
```

This demonstrates how the system handles complex reasoning tasks while managing rate limits effectively.

## How It Works

### The Reasoning Chain

1. **Digest** (for long inputs): Summarizes the prompt
2. **Plan**: Creates a structured execution plan with verification steps
3. **Critique**: Identifies potential issues and logical risks
4. **Fix**: Refines the plan based on critique feedback
5. **Execute**: Implements the plan to generate the response
6. **Score**: Evaluates using JSON format with multiple dimensions
7. **Improve**: Enhances the response if logic score is low

### Scoring Methodology

```json
{
  "clarity": 8,
  "logic": 3,
  "actionability": 7,
  "feedback": "Response is clear but contains logical inconsistencies..."
}
```

**Composite Score**: `(clarity × 2 + logic × 4 + actionability × 2) ÷ 8 × 10`

**Improvement Triggers**: 
- Composite score < 80, OR
- Logic score < 6

### Rate Limiting Protection

- **Smart detection**: Recognizes 429 errors and quota limits
- **Intelligent backoff**: Uses API-suggested retry delays
- **Proactive prevention**: Estimates token usage before requests
- **User controls**: Configurable delays and retry behavior

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google AI](https://aistudio.google.com/)
- Uses the Gemma 27B model for reasoning
- Inspired by chain-of-thought prompting research
