#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/trukhinmaksim/src')


# In[2]:


import numpy as np
import requests
from random import random
from time import sleep, time
import json


# In[3]:


print(requests.get("http://google.com"))


# In[4]:


from src.utils.CacheAdapter import JSONAdapter, JSONMultiFileAdapter
from src.utils.DatasetManager import ProjectsDatasetManager
from src.data_processing.scan_csv_files import UsersCollection
from src.utils.DatabaseConnect import DatabaseConnect

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
print(projectsCollection)


# In[5]:


# In[6]:


def flatternData(data : dict[str, list]) -> np.array(dict):
    # takes in data in form of dict, where each key is a user id and each value is a list of that user's projects
    # returns just flat list of these projects 
    result = []

    for projectsArray in data.values():
        for project in projectsArray:
            result.append(project)

    return result


# In[11]:


# Validators are used to filter data by quality, 
# for example, I can take only those project, that has long description, readme file and many stars

def projectDataIsGood(projectData):
    # filters good data (has description and both topics and language)
    try:
        return all((
            projectData,
            projectData["description"].count(" ") >= 2, # at least 2 spaces (hoping to find at least 3 words in the description)
            (len(projectData["topics"]) or projectData["language"])
        ))
    except KeyError:
        return False


USERS_NUMBER_TO_SCAN = 50

def extractScannedUsers(data):
    return list(data.keys())

cacheFileName = "cache__02-04-2025__(good)_{0}.json"

adapter = JSONMultiFileAdapter(cacheFileName, 1150)
#adapter = JSONAdapter(cacheFileName)

ProjectsDatasetManager.usersCollection = usersCollection
ProjectsDatasetManager.projectsCollection = projectsCollection
ProjectsDatasetManager.translatorServers = ["http://54.90.185.243:8000/", "http://52.91.234.245:8000/"]
manager = ProjectsDatasetManager(USERS_NUMBER_TO_SCAN, validate = projectDataIsGood, cacheAdapter = adapter)


# In[8]:


print(manager.translateText("你好", 3))


# In[9]:


with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", encoding = "utf-8") as file:
    manager.ignoreUsers(json.load(file))

#print(manager.ignoredUsers)


# In[12]:


counter = 0
startPoint = time()

try:
    for i in range(1150, 1200): #60, 70
        manager.fromDB()
        manager.preprocess()

        #print([*manager.data.items()][:5])
        #print("\n")
        scanned = extractScannedUsers(manager.data)
        manager.ignoreUsers(scanned)

        #adapter.collectionName = cacheFileName.format(i)
        #print(adapter.collectionName)
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
    print("\nException encountered!\n")
    print("\nAmount of users to ignore:")
    print(len([*manager.ignoredUsers]))
    print("Writing into file")

    with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", "w", encoding = "utf-8") as file:
        json.dump([*manager.ignoredUsers], fp = file, ensure_ascii=False, indent=4)

    raise exp


# In[ ]:


#print([*manager.ignoredUsers][220:230])
print("\nAmount of users to ignore:")
print(len([*manager.ignoredUsers]))
print("Writing into file")

with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", "w", encoding = "utf-8") as file:
    json.dump([*manager.ignoredUsers], fp = file, ensure_ascii=False, indent=4)


# In[ ]:

with open("/home/trukhinmaksim/src/logs/ignoreUsers(good).json", encoding = "utf-8") as file:
    data = json.load(fp = file)

print(len(set(data)))

"""
data += ignore

print(len(set(data)))
with open("/home/trukhinmaksim/src/logs/ignoreUsers.json", "w", encoding = "utf-8") as file:
    json.dump(data, fp = file)
"""


# In[ ]:




