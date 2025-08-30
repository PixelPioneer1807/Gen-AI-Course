
import requests

def generate_completion(message):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer euri-01fd20dad34546155d4e7ac30a637a3c9ae15c294d5f67528f9816615ae61211"
    }
    payload = {
        "messages": [{"role": "user", "content": message}],
        "model": "gpt-4.1-nano",
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # ✅ Handle structure without `success` key
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            print("Unexpected API response structure:")
            print(data)
            return None

    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e)
        print("Response text:", response.text)
    except Exception as e:
        print("Unexpected Error:", e)
        return None
