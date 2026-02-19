import sympy as sp
import logging
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('replay_logger')
events = []

def log_event(event):
    events.append(event)
    logger.info(event)

log_event('Replay started')

app = Flask(__name__)

@app.route('/diagnostic')
def diagnostic():
    return jsonify({"replay_state": len(events), "events": events})

# Example symbolic computation
x = sp.symbols('x')
expr = sp.integrate(sp.exp(-x**2), (x, 0, 1))
result = expr.evalf()
print(result)

if __name__ == '__main__':
    app.run(debug=True)