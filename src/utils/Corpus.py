import sys
sys.path.append('/home/trukhinmaksim/src')


import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean

from gensim.models.doc2vec import TaggedDocument

from src.utils.CacheAdapter import CacheAdapter, JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA, createTrainSetAdapter_02_04_25_GOOD
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.helpers import flatternData


class Corpus:
    # base class for every data corpus, that will be used by model
    def __init__(self):
        pass
    def __iter__(self):
        pass
    def __getitem__(self, index : int):
        pass

class CacheCorpus(Corpus):
    def __init__(self, manager, limit = float("inf")):
        self.manager = manager # manager is needed not only for interaction with adapter, but also if I want to use unpreprocessed dataset and preprocess it on the way
        self.limit = limit
        self.resetOnIter = False

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
        self.resetOnIter = False


    def __iter__(self):
        # will feed preprocessed projects data as TaggedDocument instances one by one

        i = 0
        while True:
            try:
                doc = self.adapter.load(1)[0]
                yield TaggedDocument(words = doc["tokens"], tags = doc["tags"])

                i += 1
                if i >= self.limit:
                    raise EXP_END_OF_DATA

            except EXP_END_OF_DATA:
            # no data left
                break

        i = 0
        self.adapter.reset()