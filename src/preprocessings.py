#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/trukhinmaksim/src')


# In[2]:


import numpy as np
import requests


# In[3]:


requests.get("http://google.com")


# In[4]:


from src.utils.DatabaseConnect import DatabaseConnect

# single machine setup (mongo is running here localy)
# "ip a" for ip address
MY_DATABASE_LINK = 'mongodb://10.22.50.212:27020/' #'mongodb://192.168.100.57:27020/'
# multiple mechine setup (mongo is running on another machine)
#MY_DATABASE_LINK = 'mongodb://192.168.43.78:27020/'

DatabaseConnect.DB_LINK = MY_DATABASE_LINK

projectsCollection = DatabaseConnect.mini_database.projects()
usersCollection = DatabaseConnect.mini_database.users()
print(projectsCollection)


# In[5]:


from src.utils.CacheAdapter import JSONAdapter
from src.utils.DatasetManager import ProjectsDatasetManager


# In[6]:


def flatternData(data : dict[str, list]) -> np.array(dict):
    # takes in data in form of dict, where each key is a user id and each value is a list of that user's projects
    # returns just flat list of these projects 
    result = []

    for projectsArray in data.values():
        for project in projectsArray:
            result.append(project)

    return result


# In[7]:


from random import random
from time import sleep, time
import json


# In[8]:


# Validators are used to filter data by quality, 
# for example, I can take only those project, that has long description, readme file and many stars

def projectDataIsSufficient(projectData):
    # filters sufficient data (has description and one(or both) of topics or language)
    return (projectData and projectData["description"] and (len(projectData["topics"]) or projectData["language"]))


USERS_NUMBER_TO_SCAN = 25

def extractScannedUsers(data):
    return list(data.keys())

counter = 0

cacheFileName = "cache__31-03-2025__(sufficient)_{0}.json"

adapter = JSONAdapter(cacheFileName)

ProjectsDatasetManager.usersCollection = usersCollection
ProjectsDatasetManager.projectsCollection = projectsCollection
ProjectsDatasetManager.translatorServers = ["http://54.90.185.243:8000/", "http://52.91.234.245:8000/"]
manager = ProjectsDatasetManager(USERS_NUMBER_TO_SCAN, validate = projectDataIsSufficient, cacheAdapter = adapter)


# In[9]:


print(manager.translateText("你好", 3))


# In[10]:


with open("/home/trukhinmaksim/src/logs/ignoreUsers.json", encoding = "utf-8") as file:
    manager.ignoreUsers(json.load(file))

#print(manager.ignoredUsers)


# In[11]:


startPoint = time()

for i in range(60, 80): #60, 70
    manager.fromDB()
    manager.preprocess()

    #print(len(manager.data))
    scanned = extractScannedUsers(manager.data)
    manager.ignoreUsers(scanned)

    adapter.collectionName = cacheFileName.format(i)
    #print(adapter.collectionName)
    adapter.save(manager.data)

    counter += len(flatternData(manager.data))

    manager.clearData()

    sleepTime = random() * 15
    print(f"Scanned {USERS_NUMBER_TO_SCAN} users. Sleeping {sleepTime}")
    sleep(sleepTime)

endPoint = time()
print(f"Total scanned: {counter} projects")
print(f"Time spent: {endPoint - startPoint} s")


# In[12]:


#print([*manager.ignoredUsers][220:230])

print(len([*manager.ignoredUsers]))

import json

with open("/home/trukhinmaksim/src/logs/ignoreUsers.json", "w", encoding = "utf-8") as file:
    json.dump([*manager.ignoredUsers], fp = file)


# In[13]:


import json
with open("/home/trukhinmaksim/src/logs/ignoreUsers.json", encoding = "utf-8") as file:
    data = json.load(fp = file)

print(len(set(data)))

"""
data += ignore

print(len(set(data)))
with open("/home/trukhinmaksim/src/logs/ignoreUsers.json", "w", encoding = "utf-8") as file:
    json.dump(data, fp = file)
"""


# In[ ]:




