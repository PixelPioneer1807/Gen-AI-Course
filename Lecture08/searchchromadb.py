import os
import requests 
from chromadb import HttpClient
from dotenv import load_dotenv
load_dotenv()

EURI_API_KEY=os.getenv("EURI_API_KEY")

client= HttpClient(host="localhost",port=8000)
colletion= client.get_or_create_collection("First_practical_data")
    
def generate_embeddings(text_list):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text_list,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    embeddings = [item["embedding"] for item in response.json()["data"]]
    
    return embeddings
def search_chroma(query_text):
    query_embed=generate_embeddings(query_text) 
    result=colletion.query(query_embeddings=query_embed, include=["documents"],n_results=3)
    print(result)
    
search_chroma("Sachin and virat are two legends of indian cricket")

# Perfectly Retrieving data 