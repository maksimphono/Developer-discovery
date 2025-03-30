import os
import json

PREPROCESSED_DATA_CACHE_PATH = "/home/trukhinmaksim/src/data/cache"

EXP_END_OF_DATA = Exception("Entire dataset has been fed from cache")

class CacheAdapter:
    def __init__(self, collectionName = ""):
        self.collectionName = collectionName

    def load(self, amount = float("inf")):
        # load amount of users, that was specified, that is needed for data fragmentation
        return {}

    def save(self, data):
        return {}

class JSONAdapter(CacheAdapter):
    PREPROCESSED_DATA_CACHE_PATH = PREPROCESSED_DATA_CACHE_PATH

    @classmethod
    def default(cls):
        return cls()
    
    def load(self, amount = float("inf")):
        # will load data from JSON file, argument 'collectionName' is a file name
        if self.collectionName:
            fileName = self.collectionName
        else:
            # take the first file from the directory:
            fileName = next(os.walk(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH))[2][0]

        print(fileName)

        try:
            with open(os.path.join(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH, fileName), encoding = "utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise EXP_END_OF_DATA

    def save(self, data):
        # will write data into the predefined JSON file
        if self.collectionName:
            fileName = self.collectionName
        else:
            # take the first file from the directory:
            fileName = next(os.walk(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH))[2][0]

        with open(os.path.join(JSONAdapter.PREPROCESSED_DATA_CACHE_PATH, fileName), "w", encoding = "utf-8") as file:
            json.dump(data, fp = file)

        return data

class DBAdapter(CacheAdapter):
    def __init__(self, cacheCollection, ignoreList):
        super().__init__(self)
        self.cacheCollection = cacheCollection
        self.ignoreList = ignoreList
    
    def load(self, amount = float("inf")):
        count = amount
        cursor = self.cacheCollection.find()
        result = {}
        
        for user in cursor:
            if count <= 0: break
            if user["id"] in self.ignoreList: continue

            result[user["id"]] = user["tokenized_projects"]

        return result
