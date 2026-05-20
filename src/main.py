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
from src.utils.DatasetManager import ProjectsDatasetManager, NewDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import CacheCorpus, Factory_21_04_25_HIGH as CorpusFactory
from src.utils.Evaluator import Evaluator, UsersEvaluator, Factory as EvaluatorFactory
from src.utils.helpers import cosineSimilarity as similarity

from skopt.space import Real, Integer
from src.utils.AutoTuner import AutoTuner, Param
from src.visualize import build
#from src.Doc2Vec_model import Model
#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import TaggedDocument
from src.SBERT_model import SiameseBert as Model
from transformers import BertTokenizer
from torch import tensor, mean as t_mean, float32 as t_float32
import torch

MODEL_SAVING_PATH = "/home/trukhinmaksim/src/src/models/28-05-25_BERT.model"
RESULTS_RECORD_PATH = "/home/trukhinmaksim/src/results/19-05-25_evaluatuin.result"
TUNER_LOG_PATH = "/home/trukhinmaksim/src/logs/19-05-25_autotunning.log"
TRAINING_LOG_PATH = "/home/trukhinmaksim/src/logs/19-05-25_training.log"

# creating model

ALPHA_INIT = 0.05
ALPHA_FINAL = 0.00001

trainCorpus = None
testCorpus = None
evaluator = None

# autotunning model parameters

class M:
    def __init__(self, model):
        self.model = model
    def call(self, doc):
        return self.model.infer_vector(doc.words)

def meanAggregator(vectors):
    sum_tensor = torch.sum(torch.stack(vectors), dim=0)

    # Divide by the number of tensors to get the average.
    return sum_tensor / len(vectors)
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(text):
    en = tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=128)

    return en

def main():
    start = time()

    
    model = Model.load(MODEL_SAVING_PATH)
    model.eval()
    #prepareCorpus(model)
    #manager = NewDatasetManager(0, inputAdapter = None)
    #d2vTokenizer = lambda text: TaggedDocument(words = manager.textPreprocessing(text, False), tags = [0])
    
    #meanAggregator = lambda vectors: t_mean(tensor(vectors, dtype=t_float32, device=model.dev), dim = 0)
    """
    relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
    testCorpus = CorpusFactory.BERT.createTestCorpus()
    model.trainCorpus = testCorpus
    model.testCorpus = testCorpus
    evaluator = Evaluator(relatedAda, unrelatedAda, testCorpus)
    evaluator.setSimilarityCheck(similarity)
    model.evaluator = evaluator
    print(f"Starting evaluation process with evaluator = {evaluator}; test corp len = {len(testCorpus.workingList)}; pairs = {len(evaluator.relatedPairs)}")
    results = model.evaluate()
    print(results)
    """
    testCorpus = CorpusFactory.BERT.createTestCorpus(2)
    model.trainCorpus = testCorpus
    model.testCorpus = testCorpus

    evaluator = UsersEvaluator(aggregate = meanAggregator, tokenizer = tokenize)
    evaluator.setSimilarityCheck(similarity)
    evaluator.setModel(model)
    model.evaluator = evaluator
    results = model.evaluate()
    print(results)
    build(evaluator, "users")
    return

    try:
        # danger zone! Progress must be saved if error occure
        print("Welcome!")
        #results = model.evaluate() # {'vector_size': 230, 'window': 5, 'min_count': 15, 'epochs': 55, 'negative': 20, 'sample': 1e-05}
        print(f"\nReady to evaluate model on users data, evaluator = {repr(evaluator)}")
        evaluator.setModel(model)
        results = evaluator.evaluate()
        #v = evaluator.getPlotVectors()
        #print(v.shape)

        end = time()
        print(f"\n\nProcess completed in {(end - start) / 60} min\n")
        print(f"Found best evaluation value {results}\n")

        with open(RESULTS_RECORD_PATH, "w") as file:
            print(results, file = file)

    except Exception as exp:
        #print(f"Error occured, last best performance score was {Model.bestScore} with parameters {Model.bestParameters}\n")
        print(str(exp))
        print("Error occured")
        raise exp
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
