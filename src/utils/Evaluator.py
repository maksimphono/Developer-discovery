import sys
sys.path.append('/home/trukhinmaksim/src')

import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean
from copy import deepcopy
from scipy.stats import mannwhitneyu

from src.utils.CacheAdapter import Factory_21_04_25_HIGH as CacheFactory
from src.utils.CacheAdapter import EvaluationAdapterFactory


class Evaluator:
    def __init__(self, relatedPairsIdxAdapter, unrelatedPairsIdxAdapter, corpus, limit = np.inf):
        self.limit = limit
        self.relatedPairsIdxAdapter = relatedPairsIdxAdapter
        self.unrelatedPairsIdxAdapter = unrelatedPairsIdxAdapter
        self.corpus = corpus
        self.memorizedVectors = {}
        self.model = None
        self.similarityCheck = lambda v1, v2: 1
        self.relatedPairsSimilarities = []
        self.unrelatedPairsSimilarities = []
        self.load()

    def load(self):
        # load both sets of pairs
        try:
            self.relatedPairs = self.relatedPairsIdxAdapter.load(self.limit)
        except EXP_END_OF_DATA:
            pass
        try:
            self.unrelatedPairs = self.unrelatedPairsIdxAdapter.load(self.limit)
        except EXP_END_OF_DATA:
            pass

        # print(f"Got {len(self.relatedPairs)} related pairs")
        # print(f"Got {len(self.unrelatedPairs)} unrelated pairs")
        if self.logger != None:
            self.logger.info(f"Got {len(self.relatedPairs)} related pairs")
            self.logger.info(f"Got {len(self.unrelatedPairs)} unrelated pairs")

    def setModel(self, model):
        # set new model and clear everything, that was produced by the old model
        self.model = model
        self.memorizedVectors.clear()
        self.relatedPairsSimilarities.clear()
        self.unrelatedPairsSimilarities.clear()

    def setSimilarityCheck(self, fn):
        self.similarityCheck = fn

    def statisticalTest(self, group1, group0):
        # print(f"Using Mann-W test on group1 = {group1[:10]}... group0 = {group0[:10]}...")
        if self.logger != None: self.logger.info(f"Using Mann-W test on group1 = {group1[:10]}... group0 = {group0[:10]}...")
        u_statistic, p_value = mannwhitneyu(group1, group0, alternative = "less") # , alternative = "less"

        if p_value < 1e-150:
            return 1
        return -p_value

    def getVector(self, index):
        if index in self.memorizedVectors:
            return self.memorizedVectors[index]
        else:
            doc = self.corpus[index]
            vec = self.model.call(doc)
            self.memorizedVectors[index] = vec
            return vec

    def evaluate(self):
        # print(f"Called Evaluate.evaluate() method, model = {repr(self.model)} relatedPairs = {self.relatedPairs[:3]}...")
        if self.logger != None: self.logger.info(f"Called Evaluate.evaluate() method, model = {repr(self.model)} relatedPairs = {self.relatedPairs[:3]}...")
        for pairs, similarities in ((self.relatedPairs, self.relatedPairsSimilarities), (self.unrelatedPairs, self.unrelatedPairsSimilarities)):
            for pair in pairs:
                item1, item2, label = [*pair.values()]
                vec1, vec2 = (self.getVector(item1), self.getVector(item2))
                simScore = self.similarityCheck(vec1, vec2)
                similarities.append(simScore)

        return self.statisticalTest(self.relatedPairsSimilarities, self.unrelatedPairsSimilarities)


class UsersEvaluator(Evaluator):
    def __init__(self, aggregate = np.mean, tokenizer = lambda s: s.split()):
        group1, group0 = CacheFactory.createProjectsEvaluationGroups()
        super().__init__(group1, group0, None, limit = 2200)

        self.aggregate = aggregate
        self.tokenizer = tokenizer

    def getUserVector(self, user):
        vectors = []
        for proj_id, text in user["projects"].items():
            if proj_id in self.memorizedVectors:
                vectors.append(self.memorizedVectors[proj_id])
            else:
                doc = self.tokenizer(text)
                vectors.append(self.model(doc))
                self.memorizedVectors[proj_id] = vectors[-1]
        
        return self.aggregate(np.array(vectors))

    def getProjectVector(self, project):
        proj_id = [*project.keys()][0]
        if proj_id in self.memorizedVectors:
            return self.memorizedVectors[proj_id]
        else:
            doc = self.tokenizer([*project.values()][0])
            vector = self.model.call(doc)
            self.memorizedVectors[proj_id] = vector
            return vector

    def getVector(self, obj):
        if "id" in obj:
            # argument is a user
            return getUserVector(obj)
        else:
            # argument is a project
            return getProjectVector(obj)


class Factory:
    @classmethod
    def createEvaluator(cls, model, corpus, similarity):
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        evaluator = Evaluator(relatedAda, unrelatedAda, corpus)
        evaluator.setModel(model)
        evaluator.setSimilarityCheck(similarity)

        return evaluator