#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample

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


class InputAdapter:
    def __init__(self):
        self.cursor = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info").find(projection = {"_id" : False, "id" : True, "proj_id" : True, "name" : True, "description" : True, "topics" : True, "language" : True})

    def reset(self, skip = 0):
        self.cursor = DatabaseConnector("mongodb://readonlyUser:cictest123456@114.212.84.247:27017/", "developer_discovery").collection("proj_info").find(projection = {"_id" : False, "id" : True, "proj_id" : True, "name" : True, "description" : True, "topics" : True, "language" : True}).skip(skip)

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

inputAdapter = InputAdapter()

collection = CacheConnector("mongodb://10.22.80.194:27020/").collection("cache_21-04-25")
outputDB = DBFlatAdapter(collection)
outputCache = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/cache_21-04-25_(HIGHQUALITY)")

manager = NewDatasetManager(
    1000,
    inputAdapter, 
    outputAdapters = [outputDB, outputCache],
    validator = projectDataIsHighQuality
)

while True:
    try:
        manager()
    except EXP_END_OF_DATA:
        break
