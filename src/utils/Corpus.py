import sys
sys.path.append('/home/trukhinmaksim/src')


import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean
from copy import deepcopy

from gensim.models.doc2vec import TaggedDocument

from src.utils.CacheAdapter import CacheAdapter, JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA, createTrainSetAdapter_02_04_25_GOOD, Factory_21_04_25_HIGH as AdapterFactory_21_04_25
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.helpers import flatternData


class Corpus:
    # base class for every data corpus, that will be used by model
    includeAllTags = True
    onlyID = False
    def __init__(self):
        pass
    def __iter__(self):
        pass
    def __getitem__(self, index : int):
        pass
    def reset(self):
        # will iterate over the corpus to the end
        for doc in self: pass

class CacheCorpus(Corpus):
    def __init__(self, manager, limit = float("inf")):
        self.manager = manager # manager is needed not only for interaction with adapter, but also if I want to use unpreprocessed dataset and preprocess it on the way
        self.limit = limit
        

    def __iter__(self):
        # will feed preprocessed projects data as TaggedDocument instances one by one
        tempStorage = [] # temporary storage for data, that was read from files

        i = 0
        while True:
            try:
                while len(tempStorage) >= 1:
                    doc = tempStorage[0]
                    yield TaggedDocument(words = doc["tokens"], tags = doc["tags"])
                    i += 1
                    if i >= self.limit:
                        raise EXP_END_OF_DATA

                    tempStorage = tempStorage[1:]

                data = flatternData(self.manager.fromCache())
                tempStorage.extend(data)

            except EXP_END_OF_DATA:
            # no data left
                break

        i = 0
        tempStorage.clear()
        self.manager.cacheAdapter.reset()

# createTrainSetAdapter_02_04_25_GOOD

class FlatCorpus(Corpus):
    # will write only projects data, without users

    def __init__(self, adapter = None, limit = float("inf")):
        if isinstance(adapter, CacheAdapter):
            self.adapter = adapter
        elif isinstance(adapter, ProjectsDatasetManager):
            # for compatibility, if I pass dataset manager instead of adapter
            self.adapter = adapter.cacheAdapter # just use it's adapter directly
        else:
            self.adapter = createTrainSetAdapter_02_04_25_GOOD()

        self.limit = limit
        self._onlyID = False

    def reset(self):
        self.adapter.reset()

    def onlyID(self, val):
        self._onlyID = bool(val)

    def __iter__(self):
        # will feed preprocessed projects data as TaggedDocument instances one by one

        i = 0
        while True:
            try:
                doc = self.adapter.load(1)[0]
                if self._onlyID:
                    yield TaggedDocument(words = doc["tokens"], tags = doc["tags"][:1]) # tags[0] is always an id
                else:
                    yield TaggedDocument(words = doc["tokens"], tags = doc["tags"])

                i += 1
                if i >= self.limit:
                    raise EXP_END_OF_DATA

            except EXP_END_OF_DATA:
            # no data left
                break

        i = 0
        self.reset()

    def __getitem__(self, _indexes):
        return [TaggedDocument(words = doc["tokens"], tags = doc["tags"]) for doc in self.adapter[_indexes]]

class MemoryCorpus(CacheCorpus):
    def __init__(self, adapter = None, limit = np.inf, includeOnlyID = True):
        self.limit = limit
        self.adapter = adapter
        self.position = 0
        self.data = tuple([TaggedDocument(words = doc["tokens"], tags = doc["tags"]) for doc in adapter.load(limit)])
        self.dataOnlyID = tuple()

        if includeOnlyID:
            self.dataOnlyID = tuple([TaggedDocument(words = doc.words, tags = doc.tags[:1]) for doc in self.data])

        self.len = len(self.data)
        self.workingList = self.data

    def reset(self):
        self.position = 0

    def onlyID(self, val):
        if val:
            if len(self.dataOnlyID) == 0: # array with only ids is empty
                self.dataOnlyID = tuple([TaggedDocument(words = doc.words, tags = doc.tags[:1]) for doc in self.data])

            self.workingList = self.dataOnlyID
        else:
            self.workingList = self.data

    def __iter__(self):
        while self.position < self.len:
            yield self.workingList[self.position]

            self.position += 1

        self.reset()

    
    def __getitem__(self, _indexes):
        return [self.workingList[i] for i in _indexes]


from src.utils.CacheAdapter import createTestSetAdapter_02_04_25_GOOD, createTrainSetAdapter_02_04_25_GOOD, createTrainSetDBadepter_02_04_25_GOOD, createTestSetDBadepter_02_04_25_GOOD

class Factory_02_04_25:
    @classmethod
    def createFlatTrainCorpus_02_04_25_GOOD(cls, limit = np.inf):
        adapter = createTrainSetAdapter_02_04_25_GOOD()
        return FlatCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTestCorpus_02_04_25_GOOD(cls, limit = np.inf):
        adapter = createTestSetAdapter_02_04_25_GOOD()
        return FlatCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTrainDBCorpus_02_04_25_GOOD(cls, limit = np.inf):
        adapter = createTrainSetDBadepter_02_04_25_GOOD()
        return FlatCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTestDBCorpus_02_04_25_GOOD(cls, limit = np.inf):
        adapter = createTestSetDBadepter_02_04_25_GOOD()
        return FlatCorpus(adapter, limit = limit)


class Factory_21_04_25_HIGH:
    AdapterFactory = AdapterFactory_21_04_25

    @classmethod
    def createNormCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createNormAdapter()
        return FlatCorpus(adapter, limit = limit)
        #return MemoryCorpus(adapter, limit = limit, includeOnlyID = False)

    @classmethod
    def createFlatCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createFlatAdapter()
        return FlatCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTrainCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createTrainSetAdapter()
        return MemoryCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTestCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createTestSetAdapter()
        return MemoryCorpus(adapter, limit = limit, includeOnlyID = False)

    @classmethod
    def createTrainDBCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createTrainSetDBadepter()
        return FlatCorpus(adapter, limit = limit)
