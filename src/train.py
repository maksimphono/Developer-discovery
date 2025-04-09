#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/trukhinmaksim/src')


# In[2]:


import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean


# In[3]:


from src.utils.CacheAdapter import JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient


# In[4]:


import gensim
from gensim.models.doc2vec import TaggedDocument


# In[5]:


from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param


# In[6]:


CACHE_FILE_NAME = "cache__02-04-2025__(good)_{0}.json"


# In[7]:


def flatternData(data : dict[str, list]) -> np.array(dict):
    # takes in data in form of dict, where each key is a user id and each value is a list of that user's projects
    # returns just flat list of these projects 
    result = []

    for projectsArray in data.values():
        for project in projectsArray:
            result.append(project)

    return result


# In[8]:


# using adapter to load data from the cache files

# TODO: place implementation of the 'Corpus' class into a separate file
class Corpus:
    # base class for every data corpus, that will be used by model
    def __init__(self):
        pass
    def __iter__(self):
        pass
    def __getitem__(self, index : int):
        pass

class CacheCorpus(Corpus):
    def __init__(self, manager, cacheFileNameTemplate = CACHE_FILE_NAME, limit = float("inf")):
        self.cacheFileNameTemplate = cacheFileNameTemplate
        self.manager = manager # manager is needed not only for interaction with adapter, but also if I want to use unpreprocessed dataset and preprocess it on the way
        self.limit = limit
        self.resetOnIter = False

    #def __iter__(self):
    #    return self

    def __iter__(self):
        # will feed preprocessed projects data as TaggedDocument instances one by one
        cacheFileName = self.cacheFileNameTemplate
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

                #self.manager.cacheAdapter.collectionName = cacheFileName.format(i)
                data = flatternData(self.manager.fromCache())
                tempStorage.extend(data)

            except EXP_END_OF_DATA:
            # no data left
                break

        i = 0
        tempStorage.clear()
        self.manager.cacheAdapter.reset()


# In[9]:


import logging
logging.basicConfig(
    filename="/home/trukhinmaksim/src/logs/09-04-25_autotunning.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


# In[10]:


class EXP_CORPUS_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.corpus' object must be an iterable structure, inherited from 'Corpus' class!")

class EXP_MANAGER_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.manager' object must be a DatasetManager instance!")


class Model(gensim.models.doc2vec.Doc2Vec):
    manager = None
    corpus = None
    bestParameters = None
    bestScore = 0

    @classmethod
    def create(cls, **kwargs):
        model = Model(
                vector_size = VECTOR_SIZE,
                dm_dbow_mode = "DM", 
                alpha_init = ALPHA_INIT,
                alpha_final = ALPHA_FINAL,
                **kwargs
            )
        cls.manager.cacheAdapter.reset()
        cls.manager.clearData()
        model.corpus = cls.corpus

        return model
    
    def __init__(self, dm_dbow_mode = "DM", pretrain_w2v = False, alpha_init = 0.05, alpha_final = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = None # corpus is an iterator(iterable class object), that will be used in "train" method of Doc2Vec model for data extraction
        self.alphaInit = alpha_init
        self.alphaFinal = alpha_final
        self.dmDbowMode = dm_dbow_mode
        self.pretrainW2V = pretrain_w2v
    
    def train(self):
        # will build vocabulary and train the model on corpus (corpus will be fed by corpus)
        
        if not isinstance(self.corpus, Corpus): raise EXP_CORPUS_IS_NONE
        #if not isinstance(self.manager, ProjectsDatasetManager): raise EXP_MANAGER_IS_NONE

        start = time()
        self.build_vocab(self.corpus)
        logging.info(f"\nVocabulary built in {time() - start} s\n")

        if self.dmDbowMode != "DM+DBOW":
            start = time()
            super().train(
                self.corpus, 
                total_examples = self.corpus_count, 
                epochs = self.epochs,
                start_alpha = self.alphaInit,
                end_alpha = self.alphaFinal
            )
            logging.info(f"\nTraining is completed in {time() - start} s\n")
        else:
            # combine DM and DBOW
            pass

    def assess(self, sampleNum = 5, silent = False, format = "full", random_state = None):
        # simple test of model performance
        # take multiple documents from the training corpus and tries to find simillar in the dataset
        # format = "full" | "mean"

        log = lambda s: print(s) if not silent else None
        #performanceGrageScale = {50 : "Random", 60 : "Poor", 70 : "Bad", 80 : "Medium", 92 : "Optimal", 97 : "Perfect"}
        totalDocuments = self.corpus_count
        if random_state != None: randomSeed(random_state)
        indexes = sample(range(totalDocuments), sampleNum)
        if format == "full":
            stats = {}

        i = 0
        avgPerformances = []

        for doc in self.corpus:
            if i >= totalDocuments: break
            if i in indexes:
                vector = self.infer_vector(doc.words)
                sims = defaultdict(lambda: 0, self.dv.most_similar([vector], topn = totalDocuments))

                log(f"Assessing document {i} ({doc.tags}). Similarities by tags:")
                if format == "full":
                    stats[i] = {
                        "similarities by tags" : {},
                        "average" : 0
                    }

                for tag in doc.tags:
                    if format == "full":
                        stats[i]["similarities by tags"][tag] = sims[tag]
                    log(f"  {tag} : {sims[tag]}")

                avgPerformances.append(mean([sims[tag] for tag in doc.tags]))
                log(f"\n  Average similarity value: {avgPerformances[-1]}\n")
                if format == "full":
                    stats[i]["average"] = avgPerformances[-1]
            i += 1

        if format == "full":
            stats["Average accuracy"] = mean(avgPerformances)
            log(f"Average accuracy: {stats['Average accuracy']}")

            return stats
        else:
            return mean(avgPerformances)

    def evaluate(self):
        # will train the model on upon-selected set of parameters and test it's performance
        self.train()

        result = self.assess(6, silent = True, format = "mean", random_state = 42)

        if Model.bestScore < result:
            Model.bestScore = result
            Model.bestParameters = {
                "min_count" : self.min_count,
                "epochs" : self.epochs,
                "negative" : self.negative,
                "sample" : self.sample,
                "window" : self.window
            }

        return result


# In[11]:


adapter = JSONMultiFileAdapter(CACHE_FILE_NAME)
manager = ProjectsDatasetManager(50, cacheAdapter = adapter)
corpus = CacheCorpus(manager)


# In[12]:


# creating model

VECTOR_SIZE = 7
EPOCHS_NUMBER = 1
WORD_MIN_COUNT = 5
WINDOW_SIZE = 7
NEGATIVE_SAMPLES_AMOUNT = 6
SUBSAMPLING_THRESHOLD = 1e-5
ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001
DM_DBOW_MODE = "DM" # "DBOW" "DM+DBOW"
"""
# finetunning is done by twicking model parameters
model = Model(
    vector_size =  VECTOR_SIZE, 
    window =       WINDOW_SIZE, 
    min_count =    WORD_MIN_COUNT, 
    epochs =       EPOCHS_NUMBER, 
    dm_dbow_mode = DM_DBOW_MODE,
    negative =     NEGATIVE_SAMPLES_AMOUNT,
    sample =       SUBSAMPLING_THRESHOLD,
    alpha_init =   ALPHA_INIT,
    alpha_final =  ALPHA_FINAL
)
model.corpus = CacheCorpus(manager, limit = 50)
#model.assess()
#model.train()
print(model.corpus_count)
"""
#model.build_vocab(documentsCorpus)
#model.train(documentsCorpus, total_examples = model.corpus_count, epochs = model.epochs)


# In[13]:


# autotunning model parameters

def createModel(**kwargs):
    model = Model(
                vector_size = VECTOR_SIZE,
                dm_dbow_mode = "DM", 
                alpha_init = ALPHA_INIT,
                alpha_final = ALPHA_FINAL,
                **kwargs
            )
    manager.cacheAdapter.reset()
    manager.clearData()
    model.corpus = corpus

    logging.info(f"\nModel created with parameters {kwargs}\n")

    return model

def main():
    logging.info("Welcome!")
    start = time()
    parameters = [
        Param(_name = "window",    _type = Integer,  _range = (5, 10),      _initial = 7),
        Param(_name = "min_count", _type = Integer,  _range = (7, 13),      _initial = 7),
        Param(_name = "epochs",    _type = Integer,  _range = (25, 45),     _initial = 25),
        Param(_name = "negative",  _type = Integer,  _range = (5, 11),      _initial = 5),
        Param(_name = "sample",    _type = Real,     _range = (1e-6, 1e-5), _initial = 1e-5),
    ]

    tuner = AutoTuner(createModel, parameters)

    logging.info(f"\nAutotuner object created successfully with parameters: {[p.name for p in parameters]}\n")
    logging.info("Starting process of autotunning...\n")

    results = tuner.tune()

    end = time()
    logging.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
    logging.info(f"Found best evaluation value {results.fun} with parameters: {results.x}\n")

    with open("/home/trukhinmaksim/src/results/09-04-25_evaluatuin.result", "w") as file:
        print(results, file = file)

try:
    main()
    exit(0)
except Exception as exp:
    logging.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
    logging.error(str(exp))
    exit(1)


# In[14]:


"""
model = createModel(
    window =       WINDOW_SIZE, 
    min_count =    WORD_MIN_COUNT, 
    epochs =       EPOCHS_NUMBER, 
    negative =     NEGATIVE_SAMPLES_AMOUNT,
    sample =       SUBSAMPLING_THRESHOLD,
    )
model.evaluate()
"""


# In[ ]:




