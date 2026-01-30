#!/usr/bin/env python3
import sys
import webbrowser
import uvicorn

from src.prompt_generation import run_training_loop

if __name__ == "__main__":
    if any(arg.lower() == "train" for arg in sys.argv[1:]):
        run_training_loop()
        sys.exit(0)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
