import requests, json

# Hent featurelisten direkte fra API'en
schema = requests.get("http://127.0.0.1:8000/predict/schema").json()

# Lav en skabelon med 0 som default for alle features
template = {f: 0 for f in schema["feature_order"]}

# Gem til payload.json
json.dump(template, open("user_input.json", "w"), indent=2)

print("✅ payload.json created with features:", len(template))
print("You can now open payload.json and fill in realistic values.")
