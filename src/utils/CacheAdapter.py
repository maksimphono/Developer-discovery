import os
import json
from itertools import islice

PREPROCESSED_DATA_CACHE_PATH = "/home/trukhinmaksim/src/data/train_02-04-25"

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

class FlatAdapter(CacheAdapter):
    # items (json objects) are written on each line of that file, one line = one json object
    def __init__(self, collectionName = "", *args, **kwargs):
        super().__init__(collectionName, *args, **kwargs)
        self.readFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), encoding = "utf-8")
        self.writeFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "a+", encoding = "utf-8")

    def __del__(self):
        self.readFp.close()
        self.writeFp.close()
        super().__del__()

    def reset():
        self.readFp.close()
        self.writeFp.close()
        self.readFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), encoding = "utf-8")
        self.writeFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "a+", encoding = "utf-8")

    def load(self, amount = 25):
        docs = []

        for i in range(amount):
            doc = json.loads(self.readFp.readline())

            if len(doc) == 0: # empty object in the line
                raise EXP_END_OF_DATA

            docs.append(doc)

        return docs

    def save(self, data):
        for doc in data:
            json.dump(doc, fp = self.writeFp, ensure_ascii = False, separators=(',', ':'))
            print("", file = self.writeFp)

        return data


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
            json.dump(data, readFp = file)

        return data

class JSONMultiFileAdapter(JSONAdapter):
    # adapter, that is capable of reading certain amount of items from multiple files
    # basically, it must read files one by one, storing data it read to the temporary storage, then as amount of data exceeds storage capacity -> return it within load method and try to fill the storage again

    def __init__(self, baseName, saveCounter = 0, loadCounter = 0):
        super().__init__("")
        self.baseName = baseName # 'baseName' must be a string, containing "{0}", so "str.format(n)" function can be applied
        self.saveCounter = saveCounter
        self.loadCounter = loadCounter
        self.tempStorage = dict()

    def reset(self):
        self.saveCounter = 0
        self.loadCounter = 0
        self.tempStorage.clear()

    def load(self, amount = 25, state = {"counter" : 0, "tempStorage" : dict()}): # state being mutable type must persist between method calls
        # method, that returns specific amount of data per time
        while len(self.tempStorage) < amount:
            # load data from the files
            try:
                self.collectionName = self.baseName.format(self.loadCounter)
                #print(f"reading {self.collectionName}")
                self.tempStorage.update(super().load()) # load the entire file content and place it to the temporal storage
                
                self.loadCounter += 1
            except EXP_END_OF_DATA:
                if len(self.tempStorage) == 0:
                    # if all files are read and temporary storage is empty -> no more data left to return
                    self.loadCounter = 0 # reset counter, get ready for reading from the first file again
                    raise EXP_END_OF_DATA # no more data left to read
                break

        if len(self.tempStorage) > amount:
            # return dictionary with specified amount of data and save the rest
            result = dict(islice(self.tempStorage.items(), amount))

            for key in result.keys():
                del self.tempStorage[key]

            return result
        else:
            # just return loaded data as it is
            result = dict(self.tempStorage)
            self.tempStorage.clear()
            return result

    def save(self, data):
        # will write the data into the file, that is pointed by the state "counter"
        # note, that the moethod doesn't care about amount of items being saved, it just saves the entire object into a file and moves the counter forward
        self.collectionName = self.baseName.format(self.saveCounter)
        super().save(data)
        self.saveCounter += 1

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


CACHE_02_04_25_GOOD_TMPLT = "/home/trukhinmaksim/src/data/cache_02-04-25/cache__02-04-2025__(good)_{0}.json"


def createAdapter_02_04_25_GOOD(*args, **kwargs):
    return JSONMultiFileAdapter(baseName = CACHE_02_04_25_GOOD_TMPLT, *args, **kwargs)