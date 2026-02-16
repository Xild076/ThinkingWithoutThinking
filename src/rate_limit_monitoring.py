import pandas as pd


RATES = {
    "gemma": {
        "rpm": 30,
        "rpd": 14400,
        "tpm": 15000
    },
    "gptoss120b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
    },
    "nemotron": {
        "rpm": 40,
        "rpd": -1, # No known limit
        "tpm": -1, # No known limit
    }
}

