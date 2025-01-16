import json 
import base64
import requests

REQUIRED_ITEMS_TYPES = ["file", "dir"]

TOCKEN = ""
with open("./github_tocken.py") as file:
    TOCKEN = file.read().replace("\n", "")

def readFile(fileData):
    return base64.b64decode(fileData).decode('utf-8')

def traverseRepo(itemsList):
    for item in itemsList:
        if item["type"] == "file":
            content = readFile(item["content"])
        elif item["type"] == "directory":
            traverseRepo(item["contents"])

def requestRepositoryData(parameters):
    owner = parameters['owner']
    repo = parameters['repo']
    headers = {"Authorization": f"token {TOCKEN}"}

    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        items = filter(lambda item: item["type"] in REQUIRED_ITEMS_TYPES, response.json())
        traverseRepo(items)
        #print(list(data))
        #with open("res.json", "w", encoding = "utf-8") as file:
            #json.dump(list(data), file, ensure_ascii = True)
            #content = base64.b64decode(data[0]).decode('utf-8')
        #print(content)
    else:
        print(f"Failed to fetch file: {response.status_code}")


def main():
    OWNER = "TabbyML"
    REPO = "tabby"

    # Replace with your token and repo details
    
    print("Req: ", url, headers)
    



