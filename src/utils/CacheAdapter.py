import os
import json

PREPROCESSED_DATA_CACHE_PATH = "/home/trukhinmaksim/src/data/cache_31-03-25"

class EXP_END_OF_DATA(Exception):
    pass

#EXP_END_OF_DATA = MyExp("Entire dataset has been fed from cache")

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

        #print(fileName)

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

class JSONMultiFileAdapter(JSONAdapter):
    # adapter, that is capable of reading certain amount of items from multiple files
    # basically, it must read files one by one, storing data it read to the temporary storage, then as amount of data exceeds storage capacity -> return it within load method and try to fill the storage again

    def __init__(self, baseName):
        super().__init__("")
        self.baseName = baseName

    def load(self, amount = 25, state = {"counter" : 0, "tempStorage" : list()}): # state being mutable type must persist between method calls
        # method, that returns specific amount of data per time
        while len(state["tempStorage"]) < amount:
            # load data from the files
            try:
                self.collectionName = self.baseName.format(state["counter"])
                state["tempStorage"].extend(super().load()) # load the entire file content and place it to the temporal storage
                state["counter"] += 1
            except EXP_END_OF_DATA:
                break

        result = state["tempStorage"]
        state["tempStorage"] = state["tempStorage"]

        return result

    def save(self, data, state = {"counter" : 0}):
        # will write the data into the file, that is pointed by the state "counter"
        # note, that the moethod doesn't care about amount of items being saved, it just saves the entire object into a file and moves the counter forward
        self.collectionName = self.baseName.format(state["counter"])
        super().save(data)
        state["counter"] += 1

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
