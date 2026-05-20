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

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.SBERT_model import SiameseRoBerta as Model #SiameseBert as Model
from src.utils.helpers import cosineSimilarity as similarity # eucledianDistance as similarity

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/27-05-25_RoBERTa.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/27-05-25_roberta_evaluation.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/27-05-25_roberta_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/27-05-25_roberta_training.log"

# creating model

ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

trainCorpus = None
testCorpus = None
evaluator = None

def createModel(**kwargs):
    global trainCorpus, testCorpus, evaluator
    model = Model.create(
        epochs = 8,
        batchSize = 32,
        **kwargs
    )

    if trainCorpus == None:
        #trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
        trainCorpus = CorpusFactory.RoBERTa.createTrainPairsCorpus(1538100) #1538100
    if testCorpus == None:
        testCorpus = CorpusFactory.RoBERTa.createTestPairsCorpus(271436)
    if evaluator == None:
        pass

    trainCorpus.reset()
    testCorpus.reset()
    model.setTrainCorpus(trainCorpus)
    model.setTestCorpus(testCorpus)
    model.evaluator = evaluator
    print(f"Length of loaded train pairs corpus = {len(trainCorpus)}")

    return model

def saveModel(model):
    #cTr = model.trainCorpus
    #cTs = model.testCorpus
    model.trainCorpus = None
    model.testCorpus = None
    model.save(MODEL_SAVING_PATH)

# autotunning model parameters

def main():
    global evaluator
    start = time()
    model = createModel()

    try:
        # danger zone! Progress must be saved if error occure
        model.logger.info("Welcome!")
        #model.logger.info(f"\nAutotuner object created successfully with parameters: {[p.name for p in parameters]}\n")
        model.logger.info("Starting process of model training...\n")

        results = model.trainMe()
        model.trainCorpus.clear()
        model.testCorpus.clear()
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        testCorpus = CorpusFactory.RoBERTa.createTestCorpus()
        evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
        evaluator.setSimilarityCheck(similarity)
        model.evaluator = evaluator
        print(f"Starting evaluation process with evaluator = {evaluator}; test corp len = {len(testCorpus.workingList)}; pairs = {len(evaluator.relatedPairs)}")
        results = model.evaluate()

        end = time()
        model.logger.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
        model.logger.info(f"Found best evaluation value {results}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)
    
    except Exception as exp:
        model.logger.error(str(exp))
        print("Error occured")
        raise exp
        exit(1)

    finally:
        saveModel(model) # saving model upon completion or in case of error
        pass

if __name__ == "__main__":    
    Model.configLogger(TRAINING_LOG_PATH)

    main()
    exit(0)
