import numpy as np
import pandas as pd
import sympy as sp

def process_chunks(chunks, window=5, z_thresh=2):
    accumulators = {}
    anomalies = 0
    buffers = {}
    for chunk in chunks:
        for entity, val in chunk.items():
            if entity not in accumulators:
                accumulators[entity] = {'sum': 0.0, 'count': 0}
            acc = accumulators[entity]
            acc['sum'] += val
            acc['count'] += 1
            buf = buffers.setdefault(entity, [])
            buf.append(val)
            if len(buf) > window:
                buf.pop(0)
            if len(buf) == window:
                arr = np.array(buf)
                mean = arr.mean()
                std = arr.std(ddof=0)
                if std > 0:
                    z = abs(val - mean) / std
                    if z > z_thresh:
                        anomalies += 1
    result = anomalies
    return result

chunks = [
    {'a':5,'b':7},
    {'a':6,'b':8},
    {'a':100,'b':9},
    {'a':5.5,'b':7.5},
    {'a':5.6,'b':8.0},
    {'a':5.7,'b':8.1},
    {'a':5.8,'b':8.2},
    {'a':5.9,'b':8.3},
    {'a':6.0,'b':8.4},
    {'a':6.1,'b':8.5},
    {'a':6.2,'b':8.6},
    {'a':6.3,'b':8.7},
    {'a':6.4,'b':8.8},
    {'a':6.5,'b':8.9},
    {'a':6.6,'b':9.0}
]

result = process_chunks(chunks, window=5, z_thresh=2)
print(result)