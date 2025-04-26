#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample, random
from time import sleep
from requests import request
from json import dumps

from pymongo.errors import ServerSelectionTimeoutError, CursorNotFound

from src.utils.DatasetManager import DatasetManager, NewDatasetManager
from src.utils.DatabaseConnect import DatabaseConnector, CacheConnector
from src.utils.CacheAdapter import CacheAdapter, EXP_END_OF_DATA, DBFlatAdapter, FlatAdapter
from src.data_processing.scan_csv_files import UsersCollection
from src.data_processing.collect_projects_data import collectOneProjectData, EXP_NOT_IN_DB
from src.utils.validators import projectDataIsHighQuality

#Wang_laoshi_connector = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info")
import logging
logging.basicConfig(
    filename="/home/trukhinmaksim/src/logs/21-04-25_preprocessing.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

TATAL_SCANNED_PROJECTS = 29410005

class InputAdapter:
    def __init__(self, skip = 0):
        self.skip = skip
        if skip: print(f"Skipping {skip} documents")
        self.cursor = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info").find(projection = {"fork" : True, "name" : True, "id" : True, "language" : True, "topics" : True, "description" : True, "proj_id" : True, "_id" : False}).skip(skip)

    def reset(self, skip = 0):
        if skip: print(f"Skipping {skip} documents")
        self.cursor = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info").find(projection = {"fork" : True, "name" : True, "id" : True, "language" : True, "topics" : True, "description" : True, "proj_id" : True, "_id" : False}).skip(skip)

    def load(self, amount = 1):
        try:
            return [next(self.cursor)] 
        except StopIteration:
            self.reset()
            raise EXP_END_OF_DATA

class BlackList:
    def __init__(self):
        self.collection = CacheConnector("mongodb://10.22.80.194:27020/").collection("scanned_ids_21-04-25")

    def add(self, item : dict):
        try:
            self.collection.insert_one({"id" : item["id"]})
        except KeyError:
            raise Exception("Please specify id field of the item")

    def contains(self, _id):
        if self.collection.find_one({"id" : _id}): return True
        return False

class BLK:
    def includes(self, item):
        if item["proj_id"] == "github:cirosantilli/x86-bare-metal-examples":
            return True
        else:
            return False

    def add(self, item):
        pass

inputAdapter = InputAdapter(skip = TATAL_SCANNED_PROJECTS)

collection = CacheConnector("mongodb://10.22.90.255:27020/").collection("cache_21-04-25")
outputDB = DBFlatAdapter(collection)
outputCache = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/cache_21-04-25_(HIGHQUALITY)")

NewDatasetManager.translatorServers = ["http://18.212.85.208:8000/", "http://54.164.98.155:8000/"]

manager = NewDatasetManager(
    1000, 
    inputAdapter, 
    outputAdapters = [outputDB, outputCache],
    validator = projectDataIsHighQuality
)
manager.blackList = BLK()
manager.totalScannedProjects = TATAL_SCANNED_PROJECTS

totalScannedProjects = TATAL_SCANNED_PROJECTS

print(request("POST", url = NewDatasetManager.translatorServers[0], headers = {'Content-Type': 'application/json'}, data = dumps({"text" : "你好"}, ensure_ascii=False, indent=4)))

while True:
    try:
        manager()
    except EXP_END_OF_DATA:
        break
    except ServerSelectionTimeoutError:
        print("Connection to database lost, retrying")
        totalScannedProjects = manager.totalScannedProjects
        manager.inputAdapter.reset(totalScannedProjects)
        sleep(random() * 15)
        continue
    #except CursorNotFound:
    #    print(f"Error 'CursorNotFound', recreating the manager with totalScannedProjects = {totalScannedProjects}")
    #    manager.inputAdapter = InputAdapter(skip = totalScannedProjects)
