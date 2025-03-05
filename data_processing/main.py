from consts import *
import os
import json

users_db = {}
projects_db = []
readFilesNames = []

def createUserData(csv_line):
    csv_list = csv_line.split(",")
    return {"id" : csv_list[0], "type" : csv_list[-1][0], "projects" : list()}

def createProjectData()

def readCSVDatabase(limit = 2):
    ALLOWED_USER_TYPES = ["A", "B"]
    count = 0
    # read data about users from the the CSV file
    for root, dirs, files in os.walk(USER_PROFILES_PATH):
        for fileName in files:
            if count >= limit: break
            if not fileName in USER_PROFILES_IGNORE_LIST: 
                print("Reading file ", fileName)

                with open(os.path.join(root, fileName)) as file:
                    file.readline()
                    for userData in filter(lambda user: user["type"] in ALLOWED_USER_TYPES, map(createUserData, file.readlines())):
                        users_db[userData["id"]] = userData

                readFilesNames.append(fileName)
                count += 1

        # don't need to go through subdirectories

def readJSONDatabase(filesNamesToRead):
    for fileName in filesNamesToRead:
        with open(os.path.join(USER_PROJ_PARTICIPATE_PATH, fileName)) as file:
            usersProfiles = json.load(file)
        projects_db.extend(map(usersProfiles))

def main():
    readCSVDatabase(1)
    with open("users_db.json", "w") as file:
        json.dump(users_db, fp = file)




main()