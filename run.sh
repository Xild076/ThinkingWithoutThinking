#!/bin/bash
pip install fastapi uvicorn python-multipart
uvicorn app:app --reload --host 0.0.0.0 --port 8000
