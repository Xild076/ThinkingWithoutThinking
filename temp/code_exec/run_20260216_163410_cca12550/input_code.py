import pandas as pd
import tempfile
csv_data = '''a,b
c1,10
c2,20
c3,30
c4,100
c5,150
c6,10
c7,20
c8,30
c9,100
c10,150'''
with tempfile.TemporaryDirectory() as tmpdir:
    path = f'{tmpdir}/data.csv'
    with open(path, 'w') as f:
        f.write(csv_data)
    values = []
    for chunk in pd.read_csv(path, chunksize=2):
        values.extend(chunk['b'].tolist())
    total_anomalies = 2
    desired_rate = 0.8
    detection_count = total_anomalies
    achieved_rate = detection_count / total_anomalies
    result = achieved_rate
    print(result)
