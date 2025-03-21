from pymongo import MongoClient

url = "mongodb://localhost:27017/"

client = MongoClient(url)

db = client["characters"]

peopleCollection = db["local"]

peopleCollection.insert_one({"name" : "Alice", "age" : 34, "list" : [1,2,3,4]})

print(peopleCollection)