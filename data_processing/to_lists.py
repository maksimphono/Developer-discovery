import json
import os
from copy import deepcopy

PROJECTS_PSEUDO_DB_DIR_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/data/processed/projects_db"
USERS_PSEUDO_DB_DIR_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/data/output/users_db"
OUTPUT_DIR_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/data/processed/as_list"
MY_DB_LINK = 'mongodb://localhost:27020/'
#COLLECTED_PROJECTS_OUTPUT_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/output/projects_db"

def fromPseudoDBtoMongo(pseudoDBdirPath):
    BUFFER_SIZE = 50
    for root, dirs, files in os.walk(pseudoDBdirPath):
        for fileName in files:
            try:
                with open(os.path.join(root, fileName), "r", encoding="utf-8") as file:
                    listDB = list(json.load(file).values())

                print(f"Started processing file {fileName}, len = {len(listDB)}")
                with open(os.path.join(OUTPUT_DIR_PATH, fileName), "w", encoding="utf-8") as file:
                    json.dump(listDB, file, ensure_ascii=False, indent=4)

                print(f"Processed data from {fileName}, total : {len(listDB)}")

                #for i in range(0, len(db), BUFFER_SIZE):
                #    myDBCollection.insertMany(db[i:i + BUFFER_SIZE])

            except Exception as exp:
                print(f"Failed to update Database file for {fileName}; reason:\n{exp}\n")

        break

fromPseudoDBtoMongo(USERS_PSEUDO_DB_DIR_PATH)