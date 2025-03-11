from consts import *
import os
import json
from copy import deepcopy
import re
from pymongo import MongoClient

SCANNED_FILES_LIST_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/code/docker/container/src/data_processing/read_files_list.txt"
USER_PROFILES_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/data/user_profiles"
USER_PROJ_PARTICIPATE_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/data/user_proj_participate"
OUTPUT_DIRECTORY = "/home/trukhinmaksim/Maksim/学习/Thesis/data/output"

MY_DB_LINK = "mongodb://localhost:27020/"
USER_PROFILES_IGNORE_LIST = []
LOG_PATH = "/home/trukhinmaksim/Maksim/学习/Thesis/code/docker/container/src/data_processing/logs.txt"

class SmallDatabaseCollection(dict):
    client = None
    link = ""

    @classmethod
    def connect(cls):
        cls.client = MongoClient(cls.link)

        return cls.client.mini_database
    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            cls.client = None

    def __init__(self, collectionName, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collectionName = collectionName

    def save(self):
        cls = type(self)
        db = cls.connect()

        for data in self.values():
            db[self.collectionName].insert_one(data)

        cls.close()

class Collection_Mongo(SmallDatabaseCollection):
    link = MY_DB_LINK

class Collection_Files(SmallDatabaseCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbCounter = 0

    def save(self):            
        with open(os.path.join(OUTPUT_DIRECTORY, f"{self.collectionName}{self.dbCounter}.json"), "w", encoding="utf-8") as file:
            json.dump(list(self.values()), fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))

        self.dbCounter += 1


users_db = Collection_Mongo("users")
projects_db = Collection_Mongo("projects")
evaluation_projects_db = Collection_Mongo("evaluation_projects") # [{"proj_id" : project_1, "users" : [user_1, user_2, ...]}, {"proj_id" : project_2, "users" : [user_5, ...]}]

scannedFilesNames = []
"""
users_db: dict[dict] = {}
projects_db: dict[dict] = {}
evaluation_projects_db = [] # [{"proj_id" : project_1, "users" : [user_1, user_2, ...]}, {"proj_id" : project_2, "users" : [user_5, ...]}]

scannedFilesNames = []
"""

class Stats:
    def __init__(self, logFilePath = LOG_PATH, append = True):
        self.logs = []
        self.logFilePath = logFilePath
        self.logsAppend = append
        self.stats = {"total_user_number" : 0, "total_proj_number" : 0}

    def log(self, string):
        self.logs.append(str(string))

    def flash(self):
        with open(self.logFilePath, "a+" if self.logsAppend else "w", encoding="utf-8") as file:
            file.write("\n".join(self.logs))
            file.write("\n")

        self.logs.clear()

    def logDBinfo(self):
        self.log("Stats about databases:")
        self.log(f"\tScanned {len(users_db)} users")
        self.log(f"\tScanned {len(projects_db)} projects")
        self.stats["total_user_number"] += len(users_db)
        self.stats["total_proj_number"] += len(projects_db)

    def logFinal(self):
        self.log(f"\nTotal users number scanned: {self.stats["total_user_number"]}")
        self.log(f"Total projects number scanned: {self.stats["total_proj_number"]}")
        self.log(f"\tScanned CSV files:\n\t\t{str(ScannedFilesManager.scannedFilesNames)}")
        self.log("END")
        self.flash()


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


class ProjectDataCreator:
    attr_list = ["id"]
    default_values = ["0"] # U = unknown 

    class many:
        @classmethod
        def fromDict(cls, dictObj : dict) -> list:
            objs = []
            for project_id in dictObj["projects"].keys():
                objs.append(dict({"id" : project_id}))
            return objs
        
    @classmethod
    def projectIDfromCSVFileName(cls, fileName):
        name = fileName.replace("user_profiles_github_", "")[:-4]
        if name.count("_") > 1:
            # process this filename manually
            with open("./process_me.json", "a+", encoding="utf-8") as file:
                json.dump({"collection" : "evaluation_projects", "name" : name, "note" : "Change the name into proper id of the project"}, file, ensure_ascii=False, indent=4)

            return name
        else:
            return name.replace("_", "/")
        
    @classmethod
    def createEvaluationProject(cls, fileName):
        proj_id = cls.projectIDfromCSVFileName(fileName)

        return {"proj_id" : proj_id, "users" : list()}


def getProjectsDataFromDict(dictObj) -> list:
    objs = []
    for project_id in dictObj["projects"].keys():
        objs.append(dict({"id" : project_id}))
    return objs


def readCSVDatabase(priorityFiles = [], limit = 2) -> str:
    ALLOWED_USER_TYPES = ["A", "B"]
    count = 0
    filesNames = []
    # read data about users from the the CSV file

    def readFile(root, fileName):
        nonlocal count, filesNames
        print("Reading file ", fileName)
        filesNames.append(fileName)
        evaluationProjectObj = ProjectDataCreator.createEvaluationProject(fileName)

        with open(os.path.join(root, fileName), encoding="utf-8") as file:
            file.readline()
            lines = file.readlines()
            for userData in filter(lambda user: user["type"] in ALLOWED_USER_TYPES, map(UserDataCreator.fromCSV, lines)):
                users_db[userData["id"]] = userData

                evaluationProjectObj["users"].append(userData["id"])

        evaluation_projects_db[evaluationProjectObj["proj_id"]] = evaluationProjectObj
        ScannedFilesManager.add(fileName)
        count += 1

    # read priority files first 
    for fileName in priorityFiles:
        if count >= limit: break
        if fileName not in USER_PROFILES_IGNORE_LIST and fileName not in ScannedFilesManager.scannedFilesNames:
            readFile(USER_PROFILES_PATH, fileName)

    # read other files
    for root, dirs, files in os.walk(USER_PROFILES_PATH):
        for fileName in files:
            if count >= limit: break
            if fileName not in USER_PROFILES_IGNORE_LIST and fileName not in ScannedFilesManager.scannedFilesNames:
                readFile(root, fileName)

        # don't need to go through subdirectories
    return filesNames

def readJSONDatabase(filesNamesToRead):
    data: str = ""

    for fileName in filesNamesToRead:
        with open(os.path.join(USER_PROJ_PARTICIPATE_PATH, fileName), encoding="utf-8") as file:
            for line in file.readlines():
                data: dict = json.loads(line)
                projectsDataList: list[dict] = ProjectDataCreator.many.fromDict(data) #getProjectsDataFromDict(data)
                user_id: str = data["user_id"]
                userProjects: list[dict] = data["projects"]

                if user_id in users_db:
                    # if I encountered a user, that is already in the database, I check his projects list and if there is already data about his projects, raise an exception
                    if len(users_db[user_id]["projects"]) != 0:
                        if users_db[user_id]["projects"] == userProjects:
                            print(f"Eq, {user_id}")
                        else:
                            #print(f"nEq, {user_id}")
                            raise Exception(f"User {user_id} already has data about his projects")

                    # set the projects dictionary for that user
                    users_db[user_id]["projects"] = deepcopy(userProjects)

                    # save the project to projects database
                    for projData in projectsDataList:
                        id = projData["id"]

                        if id not in projects_db:
                            projects_db[id] = deepcopy(projData)


def saveDatabases(database_id):
    #users_db.save()
    #projects_db.save()
    evaluation_projects_db.save()
    
    users_db.clear()
    projects_db.clear()
    evaluation_projects_db.clear()

def _saveDatabases(database_id):
    # save collected users
    #with open(os.path.join(OUTPUT_DIRECTORY, f"users_db_{database_id}.json"), "w", encoding="utf-8") as file:
    #    json.dump(users_db, fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))
    # save collected projects
    #with open(os.path.join(OUTPUT_DIRECTORY, f"projects_db_{database_id}.json"), "w", encoding="utf-8") as file:
    #    json.dump(projects_db, fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))
    # save collected projects with related users for evaluation
    with open(os.path.join(OUTPUT_DIRECTORY, f"evaluation_projects_db_{database_id}.json"), "w", encoding="utf-8") as file:
        json.dump(evaluation_projects_db, fp = file, ensure_ascii=False, indent=4, separators = (",", ":"))

    users_db.clear()
    projects_db.clear()
    evaluation_projects_db.clear()


class ScannedFilesManager:
    scannedFilesNames = []
    @classmethod
    def get(cls) -> list:
        with open(SCANNED_FILES_LIST_PATH, encoding="utf-8") as file:
            cls.scannedFilesNames = list(map(lambda s: s[:-1] if s[-1] == "\n" else s, file.readlines()))

        return cls.scannedFilesNames

    @classmethod
    def update(cls):
        with open(SCANNED_FILES_LIST_PATH, "w", encoding="utf-8") as file:
            for filePath in cls.scannedFilesNames:
                print(filePath, file = file)

    @classmethod
    def add(cls, name):
        cls.scannedFilesNames.append(name)


def getScannedFiles():
    with open(SCANNED_FILES_LIST_PATH, encoding="utf-8") as file:
        return list(file.readlines())
    
#def updateScannedFiles():
#    with open(SCANNED_FILES_LIST_PATH, "a+", encoding="utf-8") as file:
#        [file.write(filePath) for filePath in scannedFilesNames[]]

def main():
    # separating scanning logic, by scanning small amount of files at once
    PORTION_AMOUNT = 2
    FILES_PER_PORTION = 2 # in total (PORTION_AMOUNT * FILES_PER_PORTION) files will be scanned
    PRIORITY_CSV_FILES = [
        "user_profiles_github_23mf_react-native-translucent-modal.csv",
        "user_profiles_github_2017398956_react-native-textinput-maxlength-fixed.csv",
        "user_profiles_github_a7ul_react-native-exception-handler.csv",
        "user_profiles_github_afollestad_material-dialogs.csv"
    ]

    stats = Stats()
    ScannedFilesManager.get()

    stats.log("\nSTART")

    for i in range(PORTION_AMOUNT):
        filesNames = readCSVDatabase(PRIORITY_CSV_FILES, FILES_PER_PORTION)
        readJSONDatabase(map(lambda fn: fn.replace(".csv", ".json"), filesNames))

        print(f"Scanned {len(users_db)} users")
        print(f"Scanned {len(projects_db)} projects")
        #stats.logDBinfo()

        saveDatabases(i)

    ScannedFilesManager.update()

    #with open("read_files_list.txt", "w") as file:
    #    file.write("\n".join(scannedFilesNames))

    stats.logFinal()


main()