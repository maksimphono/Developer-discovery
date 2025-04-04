from pymongo import MongoClient
import json
import os
from copy import deepcopy, copy
from itertools import islice, tee


CHUNK_SIZE = 500
COLLECTED_PROJECTS_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/projects_db"
#COLLECTED_PROJECTS_OUTPUT_DIR_PATH = "D:/Maksim/学习/毕业论文/data/pseudo_database/output/projects_db"

EXP_NOT_IN_DB = Exception("Not in DB")

client = MongoClient(f'mongodb://readonlyUser:cictest123456@114.212.84.247:27017/')
# Access the database
db = client.developer_discovery
# Access the collection
collectionProjects = db.proj_info

def processStarsNum(data : dict):
    result = {**data, "stars" : data["stargazers_count"]}
    del result["stargazers_count"]
    return result

def collectOneProjectData(proj_id):
    #colectedData = {"name" : "", "description" : "", "language" : "", "topics" : list()}
    processData = processStarsNum # lambda obj: obj
    rawData = collectionProjects.find_one({"proj_id" : proj_id}, {"name" : True, "description" : True, "language" : True, "topics" : True, "stargazers_count" : True, "_id" : False})

    if rawData == None:
        #print(proj_id)
        raise EXP_NOT_IN_DB

    colectedData = processData(dict(rawData))
    missing = {}
    
    for key, value in colectedData.items():
        if value == None:
            missing[key] = ""

    colectedData.update(missing)

    return colectedData


def addProjects2File(filePath, projects):
    with open(filePath, "a+", encoding="utf-8") as file:
        for project in projects:
            json.dump(project, file, indent=4, ensure_ascii=False)
            print(",", file = file)

def collectProjectsData(projectsDBFileName : str, startFromIndex = 0, useChunks = True) -> dict:
    if useChunks: chunksNum = startFromIndex // CHUNK_SIZE
    count = 0
    projects_db: dict = dict()

    def saveChunk():
        global CHUNK_SIZE
        nonlocal chunksNum, projects_db

        vals = tee(projects_db.values(), 1)[0]
        data = list(islice(vals, chunksNum * CHUNK_SIZE, min((chunksNum + 1) * CHUNK_SIZE, len(projects_db))))

        with open(f"D:/Maksim/学习/毕业论文/data/pseudo_database/output/projects_db/project_db_3_chunk_{chunksNum}.json", "w", encoding="utf-8") as file:
            #print("qwer", file = file)
            json.dump(data, file, ensure_ascii=False, indent=4)
        chunksNum += 1

        print("Chunk saved")


    with open(os.path.join(COLLECTED_PROJECTS_DIR_PATH, projectsDBFileName), encoding="utf-8") as file:
        projects_db = json.load(file)

    keys = tee(projects_db.keys())[0]
    for project_id in islice(keys, startFromIndex, len(projects_db)):
        try:
            projectData = collectOneProjectData(project_id)
            projects_db[project_id].update(projectData)
            count += 1

            if count % 100 == 0:
                print(f"Collected project amount for {projectsDBFileName}: ", count)

            if useChunks and count % CHUNK_SIZE == 0:
                print(f"Saving chunk {chunksNum}")
                saveChunk()

        except Exception as exp:
            if exp is EXP_NOT_IN_DB:
                projects_db[project_id] = {} # should remove that item from the database


    if useChunks:
        print(f"Saving chunk {chunksNum}")
        saveChunk()
    print(f"Collected project amount for {projectsDBFileName}: ", count)
    return projects_db


def main():
    for root, dirs, files in os.walk(COLLECTED_PROJECTS_DIR_PATH):
        for fileName in files[3:]:
            try:
                print(f"Reading data from {fileName}")
                db: dict = collectProjectsData(fileName)

                # Don't need now because using chunks
                #with open(os.path.join(root, fileName), "w", encoding="utf-8") as file:
                #    json.dump(list(db.values()), file, ensure_ascii=False, indent=4)

                print(f"Database file for {fileName} updated")
                del db
            except Exception as exp:
                print(f"Failed to update Database file for {fileName}; reason:\n{exp}\n")

        break

main()