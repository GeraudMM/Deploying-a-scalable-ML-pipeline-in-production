import requests

url = "https://render-deployment-example-udacity.onrender.com/predict"

data = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "Amer-Indian-Eskimo",
    "sex": "Female",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 35,
    "native_country": "Germany"
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
