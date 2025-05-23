#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/home/trukhinmaksim/src')

import numpy as np
import requests
from random import random
from time import sleep, time
import json

from src.utils.CacheAdapter import createAdapter_02_04_25_GOOD
from src.utils.DatasetManager import ProjectsDatasetManager
from src.data_processing.scan_csv_files import UsersCollection
from src.utils.DatabaseConnect import DatabaseConnect
from src.utils.helpers import flatternData
from src.utils.validators import projectDataIsGood

# single machine setup (mongo is running here localy)
# "ip a" for ip address
MY_DATABASE_LINK = 'mongodb://10.22.50.212:27020/' #'mongodb://192.168.100.57:27020/'
WL_DATABASE_LINK = 'mongodb://readonlyUser:cictest123456@114.212.84.247:27017/'
# multiple mechine setup (mongo is running on another machine)
#MY_DATABASE_LINK = 'mongodb://192.168.43.78:27020/'

#DatabaseConnect.DB_LINK = MY_DATABASE_LINK
DatabaseConnect.DB_LINK = WL_DATABASE_LINK

usersCollection = UsersCollection(10_000, ["user_profiles_github_agl_jbig2enc.csv", "user_profiles_github_airbnb_lottie-web.csv"])
projectsCollection = DatabaseConnect.developer_discovery.proj_info()
#projectsCollection = DatabaseConnect.mini_database.projects()
#usersCollection = DatabaseConnect.mini_database.users()

# test connection to Google
print(requests.get("http://google.com"))

print(projectsCollection)

def extractScannedUsers(data):
    return list(data.keys())


USERS_NUMBER_TO_SCAN = 50

adapter = createAdapter_02_04_25_GOOD(saveCounter = 1600)

ProjectsDatasetManager.usersCollection = usersCollection
ProjectsDatasetManager.projectsCollection = projectsCollection
ProjectsDatasetManager.translatorServers = ["http://54.90.185.243:8000/", "http://52.91.234.245:8000/"]
manager = ProjectsDatasetManager(USERS_NUMBER_TO_SCAN, validate = projectDataIsGood, cacheAdapter = adapter)


print(manager.translateText("你好, 欢迎！", 3))


def loadUsersIgnoreList(manager):
    with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", encoding = "utf-8") as file:
        manager.ignoreUsers(json.load(file))
    print(f"Loaded {len(manager.ignoredUsers)} users into ignore list")


def saveUsersIgnoreList(manager):
    print("\nAmount of users to ignore:")
    print(len([*manager.ignoredUsers]))
    print("Writing into file")

    with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", "w", encoding = "utf-8") as file:
        json.dump([*manager.ignoredUsers], fp = file, ensure_ascii=False, indent=4)


def main():
    loadUsersIgnoreList(manager)

    counter = 0
    startPoint = time()

    try:
        for i in range(1600, 1650): #60, 70
            manager.fromDB()
            manager.preprocess()

            #print([*manager.data.items()][:5])
            #print("\n")
            scanned = extractScannedUsers(manager.data)
            manager.ignoreUsers(scanned)

            adapter.save(manager.data)

            counter += len(flatternData(manager.data))

            manager.clearData()

            sleepTime = 3 + random() * 17
            print(f"Scanned {(i + 1) * USERS_NUMBER_TO_SCAN} users. Sleeping {sleepTime}")
            sleep(sleepTime)

        endPoint = time()
        print(f"Total scanned: {counter} projects")
        print(f"Time spent: {endPoint - startPoint} s")

    except Exception as exp:
        print("\nException encountered! Shutting down gracefully!\n")

        raise exp

    finally:
        saveUsersIgnoreList(manager)


    with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", encoding = "utf-8") as file:
        data = json.load(fp = file)

    print(len(set(data)))

if __name__ == "__main__":
    main()