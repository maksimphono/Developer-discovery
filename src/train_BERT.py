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
from src.SBERT_model import SiameseBert as Model
from src.utils.helpers import cosineSimilarity as similarity # eucledianDistance as similarity

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/17-05-25_BERT.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/17-05-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/17-05-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/17-05-25_bert_training.log"

# creating model

ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

trainCorpus = None
testCorpus = None
evaluator = None

def createModel(**kwargs):
    global trainCorpus, testCorpus, evaluator
    model = Model.create(
        epochs = 10,
        batchSize = 64,
        **kwargs
    )

    if trainCorpus == None:
        #trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
        trainCorpus = CorpusFactory.BERT.createTrainPairsCorpus()
    if testCorpus == None:
        testCorpus = CorpusFactory.BERT.createTestPairsCorpus()
    if evaluator == None:
        pass

    trainCorpus.reset()
    testCorpus.reset()
    model.setTrainCorpus(trainCorpus)
    model.setTestCorpus(testCorpus)
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
        testCorpus = CorpusFactory.BERT.createTestCorpus()
        evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
        evaluator.setSimilarityCheck(similarity)
        model.evaluator = evaluator
        model.evaluate()

        end = time()
        model.logger.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
        model.logger.info(f"Found best evaluation value {results}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)
    
    except Exception as exp:
        #model.logger.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        model.logger.error(str(exp))
        print("Error occured")
        raise exp
        exit(1)

    finally:
        saveModel(model) # saving model upon completion or in case of error
        pass

if __name__ == "__main__":    
    AutoTuner.configLogger(TUNER_LOG_PATH)
    #Model.configLogger(TRAINING_LOG_PATH)

    main()
    exit(0)