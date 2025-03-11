import requests

url = "http://127.0.0.1:5000/predict"
data = {"humidity": 9, "wind_speed": 70}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
