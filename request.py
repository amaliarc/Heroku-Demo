import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'rate':4, 'age':35, 'daily_time':85})

print(r.json())
