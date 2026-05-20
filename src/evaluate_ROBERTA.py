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
from src.utils.Evaluator import Evaluator, Factory as EvaluatorFactory
from src.utils.helpers import cosineSimilarity as similarity #eucledianDistance as similarity
from src.SBERT_model import SiameseRoBerta as Model

from skopt.space import Real, Integer

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/27-05-25_RoBERTa.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/19-05-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/19-05-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/19-05-25_training.log"

# creating model

ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

trainCorpus = None
testCorpus = None
evaluator = None

def prepareCorpus(model = None):
    global trainCorpus, testCorpus, evaluator
    if trainCorpus == None:
        #trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
        trainCorpus = CorpusFactory.createFlatTrainCorpus(1, max_len=3)
    if testCorpus == None:
        testCorpus = CorpusFactory.createFlatTestCorpus()
    if evaluator == None:
        """
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
        evaluator.setSimilarityCheck(similarity)
        """
        evaluator = EvaluatorFactory.createDoc2VecEvaluator(
            similarity=similarity, 
            corpus=testCorpus, 
            model=model
        )

def createModel(**kwargs):
    global trainCorpus, testCorpus, evaluator
    model = Model(
                dm_dbow_mode = "DBOW", 
                alpha_init = ALPHA_INIT,
                alpha_final = ALPHA_FINAL,
                **kwargs
            )

    prepareCorpus(model)

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

class M:
    def __init__(self, model):
        self.model = model
    def call(self, doc, seed = 88):
        randomSeed(seed)
        np.random.seed(seed)
        return self.model.infer_vector(doc.words, epochs = 47, alpha = ALPHA_INIT, min_alpha = ALPHA_FINAL)


def main():
    start = time()
    """
    model = createModel( # 231, 'window': 9, 'min_count': 14, 'epochs': 45, 'negative': 18, 'sample': 3.767143405439004e-05
        workers = 32,
        epochs = 1,
        vector_size = 4,
        window = 1,
        min_count = 1,
    )
    """
    try:
        # danger zone! Progress must be saved if error occure
        print("Welcome!")
        #results = model.evaluate() # {'vector_size': 230, 'window': 5, 'min_count': 15, 'epochs': 55, 'negative': 20, 'sample': 1e-05}

        #print(results)
        if 1:#results != 1.0:
            #model = M(Doc2Vec.load(MODEL_SAVING_PATH))
            model = Model.load(MODEL_SAVING_PATH)
            #prepareCorpus(model)

            relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
            trainCorpus = CorpusFactory.RoBERTa.createTestCorpus(limit = 1)
            model.setTrainCorpus(trainCorpus)
            testCorpus = CorpusFactory.RoBERTa.createTestCorpus()
            #model.setTrainCorpus(trainCorpus)
            evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
            evaluator.setSimilarityCheck(similarity)
            model.evaluator = evaluator
            evaluator.setModel(model)
            print(f"Starting evaluation process with evaluator = {evaluator}; test corp len = {len(testCorpus.workingList)}; pairs = {len(evaluator.relatedPairs)}")
            results = model.evaluate()
            print(f"Found evaluation value {results}\n")

        end = time()
        print(f"\n\nProcess completed in {(end - start) / 60} min\n")
        print(f"Found best evaluation value {results}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)

    except Exception as exp:
        print(f"Error occured")
        print(str(exp))
        print("Error occured")
        exit(1)

    finally:
        #saveModel(model) # saving model upon completion or in case of error
        pass

def completeProcess(*args):
    # perform custom action upon completion
    exit(0)

if __name__ == "__main__":    
    #AutoTuner.configLogger(TUNER_LOG_PATH)
    #Model.configLogger(TRAINING_LOG_PATH)

    main()
    completeProcess()
