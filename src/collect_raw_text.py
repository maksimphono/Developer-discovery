#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample, random
from time import sleep
from requests import request
from json import dumps

from pymongo.errors import ServerSelectionTimeoutError, CursorNotFound

from src.utils.DatasetManager import DatasetManager, NewDatasetManager, RawTextDatasetManager
from src.utils.DatabaseConnect import DatabaseConnector, CacheConnector
from src.utils.CacheAdapter import CacheAdapter, EXP_END_OF_DATA, DBFlatAdapter, FlatAdapter
from src.data_processing.scan_csv_files import UsersCollection
from src.data_processing.collect_projects_data import collectOneProjectData, EXP_NOT_IN_DB
from src.utils.validators import projectDataIsHighQuality

#Wang_laoshi_connector = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info")
import logging
logging.basicConfig(
    filename="/home/trukhinmaksim/src/logs/21-04-25_raw_text_collection.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

TATAL_SCANNED_PROJECTS = 0

inputAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_21-04-25")

#collection = CacheConnector("mongodb://10.22.80.194:27020/").collection("cache_21-04-25")
#outputDB = DBFlatAdapter(collection)
outputCache = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_text_21-04-25")

manager = RawTextDatasetManager(
    1000,
    inputAdapter = inputAdapter, 
    outputAdapters = [outputCache]
)
#manager.data

#totalScannedProjects = TATAL_SCANNED_PROJECTS

#print(request("POST", url = NewDatasetManager.translatorServers[0], headers = {'Content-Type': 'application/json'}, data = dumps({"text" : "你好"}, ensure_ascii=False, indent=4)))

print("Welcome")

while True:
    try:
        manager()
    except EXP_END_OF_DATA:
        break
    except ServerSelectionTimeoutError:
        print("Connection to database lost, retrying")
        totalScannedProjects = manager.totalScannedProjects
        manager.reset(totalScannedProjects)
        sleep(random() * 15)
        continue
    #except CursorNotFound:
    #    print(f"Error 'CursorNotFound', recreating the manager with totalScannedProjects = {totalScannedProjects}")
    #    manager.inputAdapter = InputAdapter(skip = totalScannedProjects)
