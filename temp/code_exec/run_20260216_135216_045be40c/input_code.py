import requests
from bs4 import BeautifulSoup
import pandas as pd
import sympy as sp

url = 'https://example.com/firmware-test-report'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

 table = soup.find('table', {'id': 'test-summary'})
rows = table.find_all('tr')
data = []
for row in rows[1:]:
    cols = row.find_all('td')
    passed = int(cols[0].get_text())
    failed = int(cols[1].get_text())
    total = passed + failed
    data.append([passed, failed, total])

df = pd.DataFrame(data, columns=['passed', 'failed', 'total'])

passed_sym = sp.Integer(df['passed'].iloc[0])
failed_sym = sp.Integer(df['failed'].iloc[0])
total_sym = sp.Integer(df['total'].iloc[0])
safety_margin = (passed_sym - failed_sym) / total_sym
result = float(safety_margin)
print(result)
