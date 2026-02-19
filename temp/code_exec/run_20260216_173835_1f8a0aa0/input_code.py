import json
sample_data = '''
{
  "services": [
    {"name": "serviceA", "type": "synchronous", "contract": {"input": "string", "output": "int"}},
    {"name": "serviceB", "type": "asynchronous", "contract": {}},
    {"name": "serviceC", "type": "synchronous", "contract": {"request": "json", "response": "text"}}
  ]
}
'''
data = json.loads(sample_data)
sync_services = [s for s in data.get("services", []) if s.get("type") == "synchronous"]
contracts = {s["name"]: s.get("contract", {}) for s in sync_services}
result = contracts
print(result)