import json 
import base64
import requests
from github_token import TOKEN, WANLAOSHI_MONGO_URI, MY_MONGO_URI, MY_DB_NAME
from pymongo import MongoClient

FETCH_LIMIT = 10

def decodeFile(fileData : str) -> str:
    return base64.b64decode(fileData).decode('utf-8')

def traverseRepo(itemsList):
    for item in itemsList:
        if item["type"] == "file":
            content = decodeFile(item["content"])
        elif item["type"] == "dir":
            traverseRepo(item["url"])

def requestRepositoryData(parameters):
    owner = parameters['owner']
    repo = parameters['repo']
    headers = {"Authorization": f"token {TOKEN}"}

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


def mongoDBConnect(uri : str, databseName : str):
    # Connect to MongoDB and returns the database
    client = MongoClient(uri)
    # Access the database
    db = client[databseName]

    return db

def fetchReadmeFile(repo_url : str) -> str:
    url = f"{repo_url}/contents/README.md"
    response = requests.get(url, headers={"Authorization": f"token {TOKEN}"})

    if response.status_code == 200:
        content = response.json()["content"]
        return decodeFile(content)
    else:
        raise Exception("Error while fitch")

def parseProjects():
    counter = 0
    wanglaoshi_DB = mongoDBConnect(WANLAOSHI_MONGO_URI, "developer_discovery")
    my_DB = mongoDBConnect(MY_MONGO_URI, MY_DB_NAME)

    proj_info_collection = wanglaoshi_DB["proj_info"]
    proj_text_data_collection = my_DB["proj_text_data"]

    for project in proj_info_collection.find():
        if counter >= FETCH_LIMIT: break
        counter += 1
        repo_data = {
            "id" : project["id"],
            "description" : project["description"],
            "readme" : "", 
            "url" : project["url"]
        }
        try:
            readme_content = fetchReadmeFile(project["url"])
            repo_data["readme"] = readme_content
            print("readme_content")
            #proj_text_data_collection.insert_one(repo_data)
        except Exception as e:
            print(f"Error fetching readme for project {project['id']}: {str(e)}")


def main():
    # Replace with your token and repo details
    parseProjects()
    

main()