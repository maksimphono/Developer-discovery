from json import load
import requests
import base64
from time import time

TOKEN = ""
with open("/home/trukhinmaksim/src/environment.json") as file:
    TOKEN = load(fp = file)["GITHUB_TOKEN"]

def decodeFile(fileData : str) -> str:
    return base64.b64decode(fileData).decode('utf-8')

owner = "iamkun"
repo = "dayjs"

def fetchReadmeFile(repo_url : str) -> str:
    url = f"{repo_url}/contents/README.md"
    response = requests.get(url, headers={"Authorization": f"token {TOKEN}"})

    if response.status_code == 200:
        content = response.json()["content"]
        return decodeFile(content)
    else:
        raise Exception("Error while fitch")

url = f"https://api.github.com/repos/{owner}/{repo}"

start = time()
text = fetchReadmeFile(url)
print(f"Time = {time() - start}")
print(text)