from pymongo import MongoClient
import json
import os
from copy import deepcopy

PROJECTS_PSEUDO_DB_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/projects_db"
USERS_PSEUDO_DB_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/users_db"
MY_DB_LINK = 'mongodb://localhost:27020/'
#COLLECTED_PROJECTS_OUTPUT_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/output/projects_db"

client = MongoClient(MY_DB_LINK)
# Access the database
#print(client)
db = client.mini_database
# Access the collection
collectionProjects = db.projects
collectionUsers = db.users

def fromPseudoDBtoMongo(myDBCollection, pseudoDBdirPath):
    BUFFER_SIZE = 50
    for root, dirs, files in os.walk(pseudoDBdirPath):
        for fileName in files:
            try:
                with open(os.path.join(root, fileName), "r", encoding="utf-8") as file:
                    db = json.load(file).values()
                    myDBCollection.insertMany(list(db)) # just write everything to the database

                    print(f"Inserted data from {fileName}, total : {len(db)}")
                #for i in range(0, len(db), BUFFER_SIZE):
                #    myDBCollection.insertMany(db[i:i + BUFFER_SIZE])

            except Exception as exp:
                print(f"Failed to update Database file for {fileName}; reason:\n{exp}\n")

        break

fromPseudoDBtoMongo(collectionProjects, PROJECTS_PSEUDO_DB_DIR_PATH)