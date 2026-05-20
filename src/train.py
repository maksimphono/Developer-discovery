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

from src.utils.CacheAdapter import FlatAdapter, EXP_END_OF_DATA, createAdapter_02_04_25_GOOD, EvaluationAdapterFactory
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import CacheCorpus, Factory_21_04_25_HIGH as CorpusFactory
from src.utils.Evaluator import Evaluator, Factory as EvaluatorFactory
from src.utils.helpers import cosineSimilarity as similarity #eucledianDistance as similarity

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.Doc2Vec_model import Model
from gensim.models import Doc2Vec

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/27-05-25_Doc2Vec_dbow_cos.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/27-05-25_dbow_cos_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/27-05-25_dbow_cos_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/27-05-25_dbow_cos_training.log"

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
                workers = 32,
                **kwargs
            )

    if trainCorpus == None:
        #trainCorpus = CorpusFactory.createFlatTrainCorpus_02_04_25_GOOD(50)
        trainCorpus = CorpusFactory.createFlatTrainCorpus(max_len=128)
    if testCorpus == None:
        testCorpus = CorpusFactory.createFlatTestCorpus(max_len=128)
    if evaluator == None:
        evaluator = EvaluatorFactory.createDoc2VecEvaluator(
            similarity=similarity, 
            corpus=testCorpus, 
            model=model
        )
        #relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        #evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
        #evaluator.setSimilarityCheck(similarity)

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
    logger = None
    def __init__(self, model):
        self.model = model
    def call(self, doc, seed = 88):
        randomSeed(seed)
        np.random.seed(seed)
        return np.array(self.model.infer_vector(doc.words, epochs = 45, alpha = ALPHA_INIT, min_alpha = ALPHA_FINAL))


def main():
    start = time()
    parameters = [
        Param(_name = "vector_size", _type = Integer,  _range = (150, 240),   _initial = 159), # 159, 'window': 5, 'min_count': 15, 'epochs': 51, 'negative': 19, 'sample': 1e-05
        Param(_name = "window",      _type = Integer,  _range = (5, 22),      _initial = 5),
        Param(_name = "min_count",   _type = Integer,  _range = (7, 25),      _initial = 15),
        Param(_name = "epochs",      _type = Integer,  _range = (40, 60),     _initial = 51), # 40
        Param(_name = "negative",    _type = Integer,  _range = (5, 20),      _initial = 19), # 18
        Param(_name = "sample",      _type = Real,     _range = (1e-5, 1e-3), _initial = 1e-05),
    ]

    tuner = AutoTuner(createModel, parameters)

    try:
        # danger zone! Progress must be saved if error occure
        tuner.logger.info("Welcome!")
        tuner.logger.info(f"\nAutotuner object created successfully with parameters: {[p.name for p in parameters]}\n")
        tuner.logger.info("Starting process of autotunning...\n")

        results = tuner.tune(25)

        tuner.logger.info(f"Found best evaluation value {results.fun} with parameters: {results.x}\nTrying to train again")

        model = createModel(
            vector_size = results.x[0],
            window = results.x[1],
            min_count = results.x[2],
            epochs = results.x[3],
            negative = results.x[4],
            sample = results.x[5]
        )
        trainCorpus.reset()
        testCorpus.reset()
        model.evaluate()
        saveModel(model)
        M.logger = model.logger
        model = M(Doc2Vec.load(MODEL_SAVING_PATH))
        evaluator.setModel(model)
        evaluator.logger = model.logger
        tuner.logger.info(f"\nReady to evaluate model, evaluator = {repr(evaluator)}")
        results = evaluator.evaluate()

        end = time()
        tuner.logger.info(f"\n\nProcess completed in {(end - start) / 60} min\n")
        tuner.logger.info(f"Found evaluation value {results} after secondary training\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)

    except Exception as exp:
        tuner.logger.error(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        tuner.logger.error(str(exp))
        print("Error occured")
        saveModel(tuner.model)
        raise exp
        tuner.logger.info("Current model saved")
        exit(1)

    finally:
        #saveModel(tuner.model) # saving model upon completion or in case of error
        pass


def completeProcess(*args):
    # perform custom action upon completion
    exit(0)

if __name__ == "__main__":    
    AutoTuner.configLogger(TUNER_LOG_PATH)
    Model.configLogger(TRAINING_LOG_PATH)

    main()
    completeProcess()
