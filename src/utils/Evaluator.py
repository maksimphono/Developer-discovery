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


class Evaluator:
    def __init__(self, relatedPairsIdxAdapter, unrelatedPairsIdxAdapter, corpus):
        # load both sets of pairs
        try:
            self.relatedPairs = relatedPairsIdxAdapter.load(np.inf)
        except EXP_END_OF_DATA:
            pass
        try:
            self.unrelatedPairs = unrelatedPairsIdxAdapter.load(np.inf)
        except EXP_END_OF_DATA:
            pass

        self.corpus = corpus
        self.memorizedVectors = {}
        self.model = None
        self.similarityCheck = lambda v1, v2: 1
        self.relatedPairsSimilarities = []
        self.unrelatedPairsSimilarities = []

    def setModel(self, model):
        if self.model is not model:
            # set new model and clear everything, that was produced by the old model
            self.model = model
            self.memorizedVectors.clear()
            self.relatedPairsSimilarities.clear()
            self.unrelatedPairsSimilarities.clear()

    def setSimilarityCheck(self, fn):
        self.similarityCheck = fn

    def statisticalTest(self, group1, group0):
        u_statistic, p_value = mannwhitneyu(group1, group0, alternative = "less")

        return p_value

    def getVector(self, index):
        if index in self.memorizedVectors:
            return self.memorizedVectors[index]
        else:
            doc = self.corpus[[index]][0]
            vec = self.model.infer_vector(doc.words)
            self.memorizedVectors[index] = vec
            return vec

    def evaluate(self, model):
        for pairs, similarities in ((self.relatedPairs, self.relatedPairsSimilarities), (self.unrelatedPairs, self.unrelatedPairsSimilarities)):
            for index1, index2, label in pairs:
                vec1, vec2 = (self.getVector(index1), self.getVector(index2))
                simScore = self.similarityCheck(vec1, vec2)
                similarities.append(simScore)

        return self.statisticalTest(self.relatedPairsSimilarities, self.unrelatedPairsSimilarities)