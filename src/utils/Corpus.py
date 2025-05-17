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

from src.utils.CacheAdapter import CacheAdapter, FlatAdapter, JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA, createTrainSetAdapter_02_04_25_GOOD, Factory_21_04_25_HIGH as AdapterFactory_21_04_25
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.helpers import flatternData

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from torch import tensor, float as torch_float
from transformers.tokenization_utils_base import BatchEncoding

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
    def __init__(self, adapter = None, limit = np.inf, includeOnlyID = True, createDocument = lambda s: None):
        self.limit = limit
        self.adapter = adapter
        self.position = 0
        self.data = []
        self.len = len(self.data)
        self.workingList = self.data

    def reset(self):
        self.position = 0

    def clear(self):
        del self.data
        self.data = tuple()

    def __iter__(self):
        while self.position < self.len:
            yield self.workingList[self.position]

            self.position += 1

        self.reset()

    def __getitem__(self, _indexes):
        if isinstance(_indexes, int):
            return self.workingList[_indexes]
        return [self.workingList[i] for i in _indexes]


class Doc2VecCorpus(MemoryCorpus):
    def __init__(self, adapter = None, limit = np.inf, includeOnlyID = True, createDocument = lambda s: None):
        super().__init__(adapter = adapter, limit = limit, includeOnlyID = includeOnlyID)
        self.data = tuple([TaggedDocument(words = doc["tokens"], tags = doc["tags"]) for doc in adapter.load(limit)])
        self.dataOnlyID = tuple()

        if includeOnlyID:
            self.dataOnlyID = tuple([TaggedDocument(words = doc.words, tags = [i]) for i, doc in enumerate(self.data)])

        self.len = len(self.data)
        self.workingList = self.data

    def onlyID(self, val):
        if val:
            if len(self.dataOnlyID) == 0: # array with only ids is empty
                self.dataOnlyID = tuple([TaggedDocument(words = doc.words, tags = doc.tags[:1]) for doc in self.data])

            self.workingList = self.dataOnlyID
        else:
            self.workingList = self.data

class SBertCorpus(MemoryCorpus):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @classmethod
    def createTaggedDocument(cls, words : str, tags : list, *args, **kwargs):
        encoding = cls.tokenizer(words, *args, **kwargs)
        encoding["tags"] = tensor([hash(tag) for tag in tags] + [0] * (8 - len(tags)))
        encoding["tags"] = tensor([hash(tag) for tag in tags] + [0] * (8 - len(tags)))

        return encoding

    def __init__(self, adapter = None, limit = np.inf, includeOnlyID = True, max_len = 128):
        super().__init__(adapter, limit, includeOnlyID)
        self.max_len = max_len
        self.data = tuple([SBertCorpus.createTaggedDocument(words = doc["text"], tags = doc["tags"], truncation=True, padding='max_length', max_length=self.max_len) for doc in adapter.load(limit)]) # return_tensors='pt'
        self.len = len(self.data)
        self.workingList = self.data
        print("Corpus created sucessfuly")

    def __len__(self):
        return len(self.workingList)

    def __getitem__(self, index):
        return self.workingList[index]


class SBertPairsCorpus(MemoryCorpus):
    # acts like a corpus, filled with actual pairs or related and unrelated documents
    def __init__(self, adapter = None, relatedPairsAdapter = None, unrelatedPairsAdapter = None, limit = np.inf, includeOnlyID = True, max_len = 128):
        super().__init__(adapter, limit, includeOnlyID)
        self.max_len = max_len
        self.pairs = tuple([pair for pair in relatedPairsAdapter.load(np.floor(limit / 2))] + [pair for pair in unrelatedPairsAdapter.load(np.ceil(limit / 2))]) # return_tensors='pt'
        self.data = tuple([BatchEncoding(encoding) for encoding in adapter.load(np.inf)])
        self.len = len(self.data)
        self.workingList = self.pairs

    def __len__(self):
        return len(self.workingList)

    def __getitem__(self, index):
        # prepares an actual pair of documents and yields it
        index1, index2, label = tuple(self.workingList[index].values())
        doc1, doc2 = (self.data[index1], self.data[index2])

        return {
            "input_ids_1" : tensor(doc1['input_ids']),
            "input_ids_2" : tensor(doc2['input_ids']),
            "attention_mask_1" : tensor(doc1['attention_mask']),
            "attention_mask_2" : tensor(doc2['attention_mask']),
            "labels" : tensor(label, dtype=torch_float)
        }

    def clear(self):
        super().clear()
        del self.pairs
        self.pairs = tuple()

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
        return Doc2VecCorpus(adapter, limit = limit)

    @classmethod
    def createFlatTestCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createTestSetAdapter()
        return Doc2VecCorpus(adapter, limit = limit, includeOnlyID = False)

    @classmethod
    def createTrainDBCorpus(cls, limit = np.inf):
        adapter = AdapterFactory_21_04_25.createTrainSetDBadepter()
        return FlatCorpus(adapter, limit = limit)

    class BERT:
        @classmethod
        def createTrainCorpus(cls, limit = np.inf, max_len = 128):
            adapter = AdapterFactory_21_04_25.createTextTrainAdapter()
            return SBertCorpus(adapter, limit = limit, max_len = max_len)

        @classmethod
        def createTestCorpus(cls, limit = np.inf, max_len = 128):
            adapter = AdapterFactory_21_04_25.createTextTestAdapter()
            return SBertCorpus(adapter, limit = limit, max_len = max_len)

        @classmethod
        def createTrainPairsCorpus(cls, limit = np.inf):
            adapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/train_tokenized_BERT_21-04-25")
            relatedAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/train_related_pairs_idx_1538100_21-04-25")
            unrelatedAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/train_unrelated_pairs_idx_1538100_21-04-25")
            return SBertPairsCorpus(adapter, relatedPairsAdapter = relatedAdapter, unrelatedPairsAdapter = unrelatedAdapter, limit = limit)

        @classmethod
        def createTestPairsCorpus(cls, limit = np.inf):
            adapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_tokenized_BERT_21-04-25")
            relatedAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_related_pairs_idx_271436_21-04-25")
            unrelatedAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_unrelated_pairs_idx_271436_21-04-25")
            return SBertPairsCorpus(adapter, relatedPairsAdapter = relatedAdapter, unrelatedPairsAdapter = unrelatedAdapter, limit = limit)
