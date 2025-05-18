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
from gensim.models import Doc2Vec

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/10-05-25_Doc2Vec.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/10-05-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/10-05-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/10-05-25_training.log"

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
                workers = 4,
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

    model = createModel( # 159, 8, 14, 41, 13, 0.0008229870821300202
        vector_size = 159,
        window = 8,
        min_count = 14,
        epochs = 41,
        negative = 13,
        sample = 0.0008229870821300202,
    )

    try:
        # danger zone! Progress must be saved if error occure
        print("Welcome!")
        results = model.evaluate() # {'vector_size': 230, 'window': 5, 'min_count': 15, 'epochs': 55, 'negative': 20, 'sample': 1e-05}

        end = time()
        print(f"\n\nProcess completed in {(end - start) / 60} min\n")
        print(f"Found best evaluation value {results}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)

    except Exception as exp:
        tuner.logger.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        tuner.logger.error(str(exp))
        print("Error occured")
        exit(1)

    finally:
        saveModel(model) # saving model upon completion or in case of error
        pass

def completeProcess(*args):
    # perform custom action upon completion
    exit(0)

if __name__ == "__main__":    
    AutoTuner.configLogger(TUNER_LOG_PATH)
    Model.configLogger(TRAINING_LOG_PATH)

    main()
    completeProcess()