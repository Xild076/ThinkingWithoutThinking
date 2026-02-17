import requests
resp = requests.get("http://service-registry:8080/services")
data = resp.json()
sync_services = [s for s in data.get("services", []) if s.get("type") == "synchronous"]
contracts = {s["name"]: s.get("contract", {}) for s in sync_services}
result = contracts
print(result)