from consts import *
import os
import json
from copy import deepcopy

users_db: dict[dict] = {}
projects_db: dict[dict] = {}
readFilesNames = []

stats = {"user_number" : 0, "proj_number" : 0}

class UserDataCreator:
    attr_list = ["id", "type", "projects"]
    default_values = ["0", "U", list()] # U = unknown 
    @classmethod
    def createEmpty(cls):
        return dict(zip(UserDataCreator.attr_list, deepcopy(UserDataCreator.default_values)))
    @classmethod
    def fromCSV(cls, csv_line):
        csv_list = csv_line.split(",")
        return {"id" : csv_list[0], "type" : csv_list[-1][0], "projects" : dict()}
    
    @classmethod
    def fromDict(cls, dictObject):
        obj = UserDataCreator.createEmpty()

        for key, value in dictObject.items(): 
            if key in UserDataCreator.attr_list:
                obj[key] = value

        return obj


def getProjectsDataFromDict(dictObj) -> dict:
    objs = []
    for project_id in dictObj["projects"].keys():
        objs.append(dict({"id" : project_id}))
    return objs

def readCSVDatabase(priorityFiles = [], limit = 2):
    ALLOWED_USER_TYPES = ["A", "B"]
    count = 0
    # read data about users from the the CSV file

    def readFile(root, fileName):
        nonlocal count
        print("Reading file ", fileName)

        with open(os.path.join(root, fileName), encoding="utf-8") as file:
            file.readline()
            lines = file.readlines()
            for userData in filter(lambda user: user["type"] in ALLOWED_USER_TYPES, map(UserDataCreator.fromCSV, lines)):
                users_db[userData["id"]] = userData
                if len(users_db) == 40: 
                    pass # killme
        readFilesNames.append(fileName)
        count += 1

    # read priority files first 
    for fileName in priorityFiles:
        if count >= limit: break
        if fileName not in USER_PROFILES_IGNORE_LIST and fileName not in readFilesNames:
            readFile(USER_PROFILES_PATH, fileName)

    # read other files
    for root, dirs, files in os.walk(USER_PROFILES_PATH):
        for fileName in files:
            if count >= limit: break
            if fileName not in USER_PROFILES_IGNORE_LIST and fileName not in readFilesNames:
                readFile(root, fileName)

        # don't need to go through subdirectories

def readJSONDatabase(filesNamesToRead):
    data: str = ""
    usersProfiles: list = []
    for fileName in filesNamesToRead:
        with open(os.path.join(USER_PROJ_PARTICIPATE_PATH, fileName), encoding="utf-8", ) as file:
            for line in file.readlines():
                data: dict = json.loads(line)
                projectsDataList: list[dict] = getProjectsDataFromDict(data)
                user_id: str = data["user_id"]
                userProjects: list[dict] = data["projects"]

                if user_id in users_db:
                    # if I encountered a user, that is already in the database, I check his projects list and if there is already data about his projects, raise an exception
                    if len(users_db[user_id]["projects"]) != 0:
                        if users_db[user_id]["projects"] == userProjects:
                            print(f"Eq, {user_id}")
                        else:
                            print(f"nEq, {user_id}")
                        #raise Exception(f"User {user_id} already has data about his projects")

                    # set the projects dictionary for that user
                    users_db[user_id]["projects"] = deepcopy(userProjects)

                    # save the project to projects database
                    for projData in projectsDataList:
                        id = projData["id"]

                        if id not in projects_db:
                            projects_db[id] = deepcopy(projData)



def test():
    users_db["github:13627491210"] = UserDataCreator.fromCSV("github:13627491210,A\n")
    #userData = UserDataCreator.fromDict({"id" : })
    readJSONDatabase(["user_profiles_github_23mf_react-native-translucent-modal.json"])


def saveDatabases(database_id):
    with open(os.path.join(OUTPUT_DIRECTORY, f"users_db_{database_id}.json"), "w", encoding="utf-8") as file:
        json.dump(users_db, fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))
    with open(os.path.join(OUTPUT_DIRECTORY, f"project_db_{database_id}.json"), "w", encoding="utf-8") as file:
        json.dump(projects_db, fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))

    users_db.clear()
    projects_db.clear()



def main():
    NUMBER_FILES_TO_SCAN = 4
    PRIORITY_CSV_FILES = [
        "user_profiles_github_23mf_react-native-translucent-modal.csv",
        "user_profiles_github_2017398956_react-native-textinput-maxlength-fixed.csv",
        "user_profiles_github_a7ul_react-native-exception-handler.csv",
        "user_profiles_github_afollestad_material-dialogs.csv"
    ]

    for i in range(NUMBER_FILES_TO_SCAN):
        readCSVDatabase(PRIORITY_CSV_FILES, 1)
        readJSONDatabase(map(lambda fn: fn.replace(".csv", ".json"), readFilesNames))

        stats["user_number"] += len(users_db)
        stats["proj_number"] += len(projects_db)

        saveDatabases(i)

    with open("read_files_list.txt", "w") as file:
        file.write("\n".join(readFilesNames))
    print("Total number of users: {0}\nTotal number of projects: {1}".format(stats["user_number"], stats["proj_number"]))



main()