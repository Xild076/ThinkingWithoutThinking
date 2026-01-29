# Plan

System goal: simulated thinking using Gemma 3 27B Instruct + self improvement capabilities. 

## Pipeline:

Input -> Plan -> [Tools] -> Synthesize -> Output

### Plan

Make Plan -> Critique Plan -> Improve Plan -> Router

1. Make a plan to respond to the input text
2. Critique the plan to find faults
3. Improve based on critique
4. Use plan to route to tool use if necessary

### Tools

1. Web search
2. Creative idea generation
    - High temp responses for more creative ones
    - Generate 10 random ideas based on the criteria -> Choose the best idea
3. Code execute
    - Input requirements -> Makes code -> Runs code -/ (Recursively fixes code until it runs properly) /-> Collects output
4. Summarize

### Synthesize

Synthesize -> Critique Synthesis -> Improve Response

1. All the information together
2. Critique response to find faults
3. Improve based on critique

## Improvement System

Hallucination/Inaccuracy/Harm check -> Root cause analysis -> Root cause confirmation -> Prompt update -> Run system with new prompts -> Compare output with old output -> If new prompt outperforms, use new prompt, else, retain old and move on to new question.

