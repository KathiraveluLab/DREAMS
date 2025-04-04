import requests
import json

# URL where your Flask app is running
url = "http://127.0.0.1:5000/sentiments/caption"

# JSON payload
data = {
    "caption": "I am getting better #recovring"
}

# Make the POST request
response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})

# Print the result
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
