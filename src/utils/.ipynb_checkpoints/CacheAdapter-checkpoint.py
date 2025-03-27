import os
import json

PREPROCESSED_DATA_CACHE_PATH = "/home/trukhinmaksim/src/mycache"

class CacheAdapter:
    def __init__(self, collectionName = ""):
        self.collectionName = collectionName

    def load(self):
        return {}

    def save(self, data):
        return {}

class JSONAdapter(CacheAdapter):
    PREPROCESSED_DATA_CACHE_PATH = PREPROCESSED_DATA_CACHE_PATH

    @classmethod
    def default(cls):
        return cls()
    
    def load(self):
        # will load data from JSON file, argument 'collectionName' is a file name
        if self.collectionName:
            fileName = self.collectionName
        else:
            # take the first file from the directory:
            fileName = next(os.walk(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH))[2][0]

        print(fileName)

        with open(os.path.join(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH, fileName), encoding = "utf-8") as file:
            return json.load(file)

    def save(self, data):
        # will write data into the predefined JSON file
        if self.collectionName:
            fileName = self.collectionName
        else:
            # take the first file from the directory:
            fileName = next(os.walk(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH))[2][0]

        with open(os.path.join(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH, fileName), "w", encoding = "utf-8") as file:
            json.dump(usersProjects, fp = file)

        return data
