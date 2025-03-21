from pymongo import MongoClient
import json

# Replace 'localhost' and '27017' with your MongoDB host and port
client = MongoClient("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/developer_discovery")

# Access a database (replace 'mydatabase' with your database name)
db = client['developer_discovery']

# Access a collection (replace 'mycollection' with your collection name)
collection = db['events']

proj_ids = {"projects" : {}}

c = 0
for event in collection.find():
    if c > 100: break
    proj_id = event["proj_id"]
    print(proj_id)
    if proj_ids["projects"].get(proj_id) == None:
        # if the entry isn't in the dictionary yet
        proj_ids["projects"][proj_id] = {"proj_id" : event["proj_id"], "project_name" : "", "readme" : "", "description" : ""}
        c += 1

proj_ids["projects"] = list(proj_ids["projects"].values())
with open("projects_list.json", "w") as file:
    json.dump(proj_ids, file)