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

from src.utils.CacheAdapter import JSONMultiFileAdapter, EXP_END_OF_DATA, createAdapter_02_04_25_GOOD, EvaluationAdapterFactory
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import CacheCorpus, Factory_21_04_25_HIGH as CorpusFactory
from src.utils.Evaluator import Evaluator
from src.utils.helpers import cosineSimilarity as similarity

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.Doc2Vec_model import Model

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/09-05-25_Doc2Vec.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/09-05-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/09-05-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/09-05-25_training.log"

# creating model

ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

trainCorpus = None
testCorpus = None
evaluator = None


def createModel(**kwargs):
    global trainCorpus, testCorpus, evaluator
    model = Model(
                dm_dbow_mode = "DBOW", 
                alpha_init = ALPHA_INIT,
                alpha_final = ALPHA_FINAL,
                workers = 16,
                **kwargs
            )

    if trainCorpus == None:
        #trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
        trainCorpus = CorpusFactory.createFlatTrainCorpus()
    if testCorpus == None:
        testCorpus = CorpusFactory.createFlatTestCorpus()
    if evaluator == None:
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
        evaluator.setSimilarityCheck(similarity)

    trainCorpus.reset()
    testCorpus.reset()
    model.trainCorpus = trainCorpus
    model.testCorpus = testCorpus
    model.evaluator = evaluator

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
        Param(_name = "vector_size", _type = Integer,  _range = (150, 230),   _initial = 220), # 175
        Param(_name = "window",      _type = Integer,  _range = (5, 15),      _initial = 7),
        Param(_name = "min_count",   _type = Integer,  _range = (7, 15),      _initial = 14),
        Param(_name = "epochs",      _type = Integer,  _range = (35, 55),     _initial = 40), # 40
        Param(_name = "negative",    _type = Integer,  _range = (5, 20),      _initial = 18), # 18
        Param(_name = "sample",      _type = Real,     _range = (1e-5, 1e-3), _initial = 0.0009151125514672825),
    ]

    tuner = AutoTuner(createModel, parameters)

    try:
        # danger zone! Progress must be saved if error occure
        tuner.logger.info("Welcome!")
        tuner.logger.info(f"\nAutotuner object created successfully with parameters: {[p.name for p in parameters]}\n")
        tuner.logger.info("Starting process of autotunning...\n")
    
        results = tuner.tune(26)

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

def completeProcess(*args):
    # perform custom action upon completion
    exit(0)

if __name__ == "__main__":    
    AutoTuner.configLogger(TUNER_LOG_PATH)
    Model.configLogger(TRAINING_LOG_PATH)

    main()
    completeProcess()