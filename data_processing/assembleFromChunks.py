import json
import os
import re

SRC_DIRECTORY = "/run/media/trukhinmaksim/DATA/Maksim/学习/毕业论文/data/pseudo_database/output/projects_db"
OUTPUT_FILE_PATH = "/run/media/trukhinmaksim/DATA/Maksim/学习/毕业论文/data/pseudo_database/output/project_db_3.json"

def readFilesNames(namePattern = ".*"):
    fileNames = []
    
    for root, dirs, files in os.walk(SRC_DIRECTORY):
        for fileName in files:
            if re.match(namePattern, fileName):
                fileNames.append(fileName)

    return fileNames

def assemble():
    fullObj = []
    fileNames = readFilesNames()

    for fileName in sorted(fileNames, key = lambda name: int(name[19:name.index(".")])):
        print(fileName)
        #print(len(fullObj))
        with open(os.path.join(SRC_DIRECTORY, fileName), encoding="utf-8") as file:
            data = [proj for proj in json.load(file) if len(proj)]
            print(len(data))

        fullObj.extend(data)

    print(len(fullObj))
    #with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as file:
    #    json.dump(fullObj, file, indent=4, ensure_ascii=False)

assemble()