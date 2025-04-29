import os
import json
from itertools import islice
import logging

from src.utils.DatabaseConnect import CacheConnector, CacheConnector_02_04_25

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
        self.readFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "r", encoding = "utf-8")
        #self.writeFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "a+", encoding = "utf-8")

    def __del__(self):
        self.readFp.close()
        logging.info("Delete adapter")
        #self.writeFp.close()
        #super().__del__()

    def resetRead(self):
        self.readFp.close()
        #self.writeFp.close()
        self.readFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), encoding = "utf-8")
        #self.writeFp = open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "a+", encoding = "utf-8")

    def reset(self):
        self.resetRead()

    def __getitem__(self, _indexes = list()):
        # iterative approach: traverse through entire adapter in search of these indexes
        results = []
        indexes = [*_indexes] # !note: '_indexes' must be sorted

        self.reset()
        i = 0
        line = ""
        while line := self.readFp.readline():
            if len(indexes) == 0: break
            if i == indexes[0]:
                results.append(json.loads(line))
                indexes.pop(0) # move to the next index

            i += 1

        self.reset()
        return results

    def load(self, amount = 25):
        docs = []

        i = 0
        while i < amount:
            line = self.readFp.readline()

            if len(line) == 0: # empty object in the line
                if len(docs) > 0:
                    return docs
                else:
                    #self.reset()
                    raise EXP_END_OF_DATA

            docs.append(json.loads(line))
            i += 1

        return docs

    def save(self, data):
        with open(os.path.join(PREPROCESSED_DATA_CACHE_PATH, self.collectionName), "a+", encoding = "utf-8") as writeFp:
            for doc in data:
                json.dump(doc, fp = writeFp, ensure_ascii = False, separators=(',', ':'))
                writeFp.write("\n")

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

class DBFlatAdapter(CacheAdapter):
    def __init__(self, cacheCollection):
        super().__init__("")
        self.cacheCollection = cacheCollection
        self.readCursor = self.cacheCollection.find(projection = {"_id" : False, "order" : False})
        self.size = self.cacheCollection.count_documents({})

    def reset(self):
        self.readCursor = self.cacheCollection.find(projection = {"_id" : False, "order" : False})

    def __getitem__(self, indexes : list = list()):
        return list(self.cacheCollection.find({"order": {'$in': indexes}}, {"_id" : False, "order" : False}).sort("order", 1))

    def load(self, amount = 25):
        result = []

        for count in range(amount):
            try:
                doc = next(self.readCursor)
            except StopIteration:
                if len(result) > 0:
                    return result
                else:
                    #self.reset()
                    raise EXP_END_OF_DATA

            result.append(doc)

        return result


    def save(self, data):
        preparedData = []

        for doc in data:
            preparedData.append({
                "order" : self.size,
                "tokens" : list(doc["tokens"]),
                "tags" : list(doc["tags"])
            })
            self.size += 1

        self.cacheCollection.insert_many(preparedData)

        return preparedData


CACHE_02_04_25_GOOD_TMPLT = "/home/trukhinmaksim/src/data/cache_02-04-25/cache__02-04-2025__(good)_{0}.json"
TRAIN_CACHE_02_04_25_GOOD = "/home/trukhinmaksim/src/data/train_02-04-25/train_02-04-25"
TEST_CACHE_02_04_25_GOOD = "/home/trukhinmaksim/src/data/train_02-04-25/test_02-04-25"
DB_LINK = "mongodb://10.22.90.255:27020/"

#@classmethod
def createAdapter_02_04_25_GOOD(*args, **kwargs):
    return JSONMultiFileAdapter(baseName = CACHE_02_04_25_GOOD_TMPLT, *args, **kwargs)

#@classmethod
def createTrainSetAdapter_02_04_25_GOOD():
    return FlatAdapter(TRAIN_CACHE_02_04_25_GOOD)

#@classmethod
def createTestSetAdapter_02_04_25_GOOD():
    return FlatAdapter(TEST_CACHE_02_04_25_GOOD)

#@classmethod
def createNormAdapter_02_04_25_GOOD():
    return FlatAdapter("/home/trukhinmaksim/src/data/normalized_02-04-25_(good)/normalized_02-04-25_(good)")

#@classmethod
def createTrainSetDBadepter_02_04_25_GOOD():
    connector = CacheConnector_02_04_25(DB_LINK)
    collection = connector.train_02_04_25
    return DBFlatAdapter(collection)

#@classmethod
def createTestSetDBadepter_02_04_25_GOOD():
    connector = CacheConnector_02_04_25(DB_LINK)
    collection = connector.test_02_04_25
    return DBFlatAdapter(collection)

class Factory_21_04_25_HIGH:
    @classmethod
    def createNormAdapter(cls):
        return FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/normalized_21-04-25_(high)")

    @classmethod
    def createFlatAdapter(cls):
        return FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/cache_21-04-25_(high)")

    @classmethod
    def createDBadapter(cls):
        connector = CacheConnector(DB_LINK)
        collection = connector.collection("cache_21-04-25")
        return DBFlatAdapter(collection)

    @classmethod
    def createTrainSetAdapter(cls):
        return FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/train_21-04-25")

    @classmethod
    def createTestSetAdapter(cls):
        return FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_21-04-25")