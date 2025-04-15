#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/trukhinmaksim/src')

import logging
import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean

from src.utils.CacheAdapter import JSONMultiFileAdapter, EXP_END_OF_DATA, createAdapter_02_04_25_GOOD
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import CacheCorpus, Factory as CorpusFactory

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.Doc2Vec_model import Model

CACHE_FILE_NAME = "cache__02-04-2025__(good)_{0}.json"

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/15-04-25_Doc2Vec.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/15-04-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/15-04-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/15-04-25_training.log"


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
    #manager.cacheAdapter.reset()
    #manager.clearData()
    model.trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
    model.testCorpus = CorpusFactory.createFlatTestCorpus_02_04_25_GOOD(10)

    return model

def saveModel(model):
    #cTr = model.trainCorpus
    #cTs = model.testCorpus
    model.trainCorpus = None
    model.testCorpus = None
    model.save(MODEL_SAVING_PATH)

# autotunning model parameters

def main():
    start = time()
    parameters = [
        Param(_name = "window",    _type = Integer,  _range = (5, 10),      _initial = 7),
        Param(_name = "min_count", _type = Integer,  _range = (1, 3),      _initial = 1), # 7 (7, 13)
        Param(_name = "epochs",    _type = Integer,  _range = (2, 5),     _initial = 5),
        Param(_name = "negative",  _type = Integer,  _range = (5, 11),      _initial = 5),
        Param(_name = "sample",    _type = Real,     _range = (1e-6, 1e-5), _initial = 1e-6), #1e-6
    ]

    tuner = AutoTuner(createModel, parameters)

    try:
        # danger zone! Progress must be saved if error occure
        tuner.logger.info("Welcome!")
        tuner.logger.info(f"\nAutotuner object created successfully with parameters: {[p.name for p in parameters]}\n")
        tuner.logger.info("Starting process of autotunning...\n")
    
        results = tuner.tune(2)

        end = time()
        tuner.logger.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
        tuner.logger.info(f"Found best evaluation value {results.fun} with parameters: {results.x}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)
    
    except Exception as exp:
        tuner.logger.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        tuner.logger.error(str(exp))
        print("Error occured")
        exit(1)

    finally:
        saveModel(tuner.model) # saving model upon completion or in case of error

if __name__ == "__main__":    
    AutoTuner.configLogger(TUNER_LOG_PATH)
    Model.configLogger(TRAINING_LOG_PATH)

    main()
    exit(0)