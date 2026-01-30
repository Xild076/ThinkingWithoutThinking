# Proposed New Tools for ThinkingWithoutThinking Pipeline

This document outlines new tools that could enhance the pipeline's capabilities.

## 1. WikipediaSearchToolBlock

**Purpose:** Fetch factual information from Wikipedia for knowledge-intensive queries.

**Why:** DuckDuckGo search can be noisy with ads and irrelevant results. Wikipedia provides high-quality, structured factual content ideal for answering knowledge questions.

**Implementation:**
```python
class WikipediaSearchToolBlock(ToolBlock):
    details = {
        "description": "Search Wikipedia for factual information on a topic.",
        "id": "wikipedia_search_tool",
        "inputs": ["query (str): The topic or question to search for."],
        "outputs": ["result (str): Summary and relevant content from Wikipedia."]
    }
    
    def __call__(self, query: str) -> str:
        import wikipedia
        try:
            # Get summary (first paragraph)
            summary = wikipedia.summary(query, sentences=5)
            return summary
        except wikipedia.DisambiguationError as e:
            # Return first option's summary
            return wikipedia.summary(e.options[0], sentences=5)
        except wikipedia.PageError:
            return f"No Wikipedia page found for: {query}"
```

**Dependencies:** `pip install wikipedia-api`

---

## 2. MathSolverToolBlock

**Purpose:** Solve mathematical equations symbolically and numerically.

**Why:** Current Python execution is general-purpose but could benefit from a specialized math tool using SymPy for symbolic math, calculus, algebra, and equation solving.

**Implementation:**
```python
class MathSolverToolBlock(ToolBlock):
    details = {
        "description": "Solve mathematical problems including algebra, calculus, and equations.",
        "id": "math_solver_tool",
        "inputs": [
            "expression (str): Math expression or equation to solve.",
            "operation (str): Type of operation (simplify, solve, differentiate, integrate, evaluate)."
        ],
        "outputs": ["result (str): The mathematical result."]
    }
    
    def __call__(self, expression: str, operation: str = "simplify") -> str:
        from sympy import sympify, solve, diff, integrate, symbols
        from sympy.parsing.sympy_parser import parse_expr
        
        x = symbols('x')
        expr = parse_expr(expression)
        
        if operation == "simplify":
            return str(expr.simplify())
        elif operation == "solve":
            return str(solve(expr, x))
        elif operation == "differentiate":
            return str(diff(expr, x))
        elif operation == "integrate":
            return str(integrate(expr, x))
        elif operation == "evaluate":
            return str(expr.evalf())
        return str(expr)
```

**Dependencies:** `pip install sympy`

---

## 3. FileAnalyzerToolBlock

**Purpose:** Analyze uploaded files (PDFs, text files, CSVs) and extract information.

**Why:** Enables document Q&A workflows, data analysis from CSVs, and reading PDFs.

**Implementation:**
```python
class FileAnalyzerToolBlock(ToolBlock):
    details = {
        "description": "Analyze file contents including PDFs, text files, and CSVs.",
        "id": "file_analyzer_tool",
        "inputs": [
            "file_path (str): Path to the file to analyze.",
            "query (str): What to look for or extract from the file."
        ],
        "outputs": ["result (str): Relevant information extracted from the file."]
    }
    
    def __call__(self, file_path: str, query: str) -> str:
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages)
        elif ext == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            text = f"Columns: {list(df.columns)}\nShape: {df.shape}\nHead:\n{df.head().to_string()}"
        else:
            with open(file_path, 'r') as f:
                text = f.read()
        
        return f"File content (first 2000 chars):\n{text[:2000]}"
```

**Dependencies:** `pip install PyPDF2 pandas`

---

## 4. DateTimeToolBlock

**Purpose:** Handle date/time calculations, timezone conversions, and scheduling queries.

**Why:** Common user queries involve "What time is it in Tokyo?", "How many days until Christmas?", etc.

**Implementation:**
```python
class DateTimeToolBlock(ToolBlock):
    details = {
        "description": "Handle date/time queries, timezone conversions, and calculations.",
        "id": "datetime_tool",
        "inputs": [
            "query (str): The date/time question or calculation.",
            "timezone (str): Optional target timezone."
        ],
        "outputs": ["result (str): The date/time answer."]
    }
    
    def __call__(self, query: str, timezone: str = "UTC") -> str:
        from datetime import datetime, timedelta
        import pytz
        
        now = datetime.now(pytz.timezone(timezone))
        
        if "current time" in query.lower() or "what time" in query.lower():
            return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        elif "days until" in query.lower():
            # Parse target date and calculate
            return f"Date calculation for: {query}"
        else:
            return f"Current datetime: {now.isoformat()}"
```

**Dependencies:** `pip install pytz`

---

## 5. ImageGenerationToolBlock

**Purpose:** Generate images from text descriptions using DALL-E or Stable Diffusion.

**Why:** Enables creative tasks, diagram generation, and visual content creation.

**Implementation:**
```python
class ImageGenerationToolBlock(ToolBlock):
    details = {
        "description": "Generate images from text descriptions.",
        "id": "image_generation_tool",
        "inputs": ["prompt (str): Description of the image to generate."],
        "outputs": ["result (str): URL or path to the generated image."]
    }
    
    def __call__(self, prompt: str) -> str:
        from openai import OpenAI
        import os
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
```

**Dependencies:** OpenAI API key, `pip install openai`

---

## 6. WeatherToolBlock

**Purpose:** Get current weather and forecasts for locations.

**Why:** Very common user query type that requires real-time external data.

**Implementation:**
```python
class WeatherToolBlock(ToolBlock):
    details = {
        "description": "Get weather information for a location.",
        "id": "weather_tool",
        "inputs": ["location (str): City name or coordinates."],
        "outputs": ["result (str): Current weather and forecast."]
    }
    
    def __call__(self, location: str) -> str:
        import requests
        import os
        
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            return f"""Weather in {location}:
            Temperature: {data['main']['temp']}°C
            Feels like: {data['main']['feels_like']}°C
            Humidity: {data['main']['humidity']}%
            Conditions: {data['weather'][0]['description']}
            Wind: {data['wind']['speed']} m/s"""
        else:
            return f"Could not fetch weather for {location}"
```

**Dependencies:** OpenWeatherMap API key

---

## 7. TranslationToolBlock

**Purpose:** Translate text between languages.

**Why:** Enables multi-language support and translation tasks.

**Implementation:**
```python
class TranslationToolBlock(ToolBlock):
    details = {
        "description": "Translate text between languages.",
        "id": "translation_tool",
        "inputs": [
            "text (str): Text to translate.",
            "target_language (str): Target language code (e.g., 'es', 'fr', 'de').",
            "source_language (str): Source language code (auto-detect if not specified)."
        ],
        "outputs": ["result (str): Translated text."]
    }
    
    def __call__(self, text: str, target_language: str, source_language: str = "auto") -> str:
        from deep_translator import GoogleTranslator
        
        translator = GoogleTranslator(source=source_language, target=target_language)
        return translator.translate(text)
```

**Dependencies:** `pip install deep-translator`

---

## 8. CodeAnalyzerToolBlock

**Purpose:** Analyze code for bugs, security issues, and complexity metrics.

**Why:** Useful for code review tasks and debugging assistance.

**Implementation:**
```python
class CodeAnalyzerToolBlock(ToolBlock):
    details = {
        "description": "Analyze code for issues, complexity, and quality.",
        "id": "code_analyzer_tool",
        "inputs": [
            "code (str): The code to analyze.",
            "language (str): Programming language."
        ],
        "outputs": ["result (str): Analysis report including issues and metrics."]
    }
    
    def __call__(self, code: str, language: str = "python") -> str:
        import ast
        
        if language == "python":
            try:
                tree = ast.parse(code)
                # Count functions, classes, complexity
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                return f"""Code Analysis:
                - Lines: {len(code.splitlines())}
                - Functions: {len(functions)}
                - Classes: {len(classes)}
                - Syntax: Valid ✓"""
            except SyntaxError as e:
                return f"Syntax Error at line {e.lineno}: {e.msg}"
        
        return f"Analysis for {language} not yet implemented"
```

---

## Priority Recommendations

Based on common use cases and implementation complexity:

### High Priority (Implement First)
1. **WikipediaSearchToolBlock** - Low effort, high value for factual queries
2. **DateTimeToolBlock** - Simple, handles common queries
3. **MathSolverToolBlock** - High value for STEM queries

### Medium Priority
4. **WeatherToolBlock** - Common query type, requires API key
5. **TranslationToolBlock** - Easy implementation with deep-translator

### Lower Priority (Future)
6. **FileAnalyzerToolBlock** - Requires file upload infrastructure
7. **ImageGenerationToolBlock** - Costly, requires DALL-E API
8. **CodeAnalyzerToolBlock** - Overlaps with Python execution tool

---

## Implementation Notes

1. All tools should inherit from `ToolBlock` base class
2. Add retry logic similar to `WebSearchToolBlock`
3. Include proper error handling and timeouts
4. Add tools to the default tool list in `Pipeline.__init__`
5. Update the tool router prompt to understand new tool capabilities
