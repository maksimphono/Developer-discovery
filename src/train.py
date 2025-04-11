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

from src.utils.CacheAdapter import JSONMultiFileAdapter, EXP_END_OF_DATA, createAdapter_02_04_25_GOOD
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import CacheCorpus

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.Doc2Vec_model import Model

CACHE_FILE_NAME = "cache__02-04-2025__(good)_{0}.json"

# using adapter to load data from the cache files

import logging
logging.basicConfig(
    #filename="/home/trukhinmaksim/src/logs/09-04-25_autotunning.log",
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

adapter = createAdapter_02_04_25_GOOD()#JSONMultiFileAdapter(CACHE_FILE_NAME)
manager = ProjectsDatasetManager(50, cacheAdapter = adapter)
corpus = CacheCorpus(manager, 100)

# creating model

VECTOR_SIZE = 7
ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

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


# autotunning model parameters

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

    results = tuner.tune(2)

    end = time()
    logging.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
    logging.info(f"Found best evaluation value {results.fun} with parameters: {results.x}\n")

    with open("/home/trukhinmaksim/src/results/09-04-25_evaluatuin.result", "w") as file:
        print(results, file = file)


if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as exp:
        logging.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        logging.error(str(exp))
        exit(1)