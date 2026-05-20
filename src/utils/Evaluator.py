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

from src.utils.CacheAdapter import Factory_21_04_25_HIGH as CacheFactory, EXP_END_OF_DATA
from src.utils.CacheAdapter import EvaluationAdapterFactory
from src.utils.Plots import buildDistributionGraphs

from src.utils.DatasetManager import ProjectsDatasetManager, NewDatasetManager
from gensim.models.doc2vec import TaggedDocument

import torch
from transformers import BertTokenizer


class Evaluator:
    def __init__(self, relatedPairsIdxAdapter, unrelatedPairsIdxAdapter, corpus, limit = np.inf):
        self.limit = limit
        self.relatedPairsIdxAdapter = relatedPairsIdxAdapter
        self.unrelatedPairsIdxAdapter = unrelatedPairsIdxAdapter
        self.corpus = corpus
        self.memorizedVectors = {}
        self.model = None
        self.similarityCheck = lambda v1, v2: 0
        self.relatedPairsSimilarities = []
        self.unrelatedPairsSimilarities = []
        self.logger = None
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
        #print("Average: ", np.mean(group0), np.mean(group1))
        self.group1 = group1
        self.group0 = group0
        u_statistic, p_value = mannwhitneyu(group1, group0, alternative = "greater") # , alternative = "less"

        return -p_value
    
    def getPlotVectors(self):
        vecs = []
        for vec in self.memorizedVectors.values():
            vecs.append(vec.squeeze(0).cpu().numpy())
            if len(vecs) >= 600:
                break

        return np.array(vecs)

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
        if self.logger != None: self.logger.info(f"Called Evaluate.evaluate() method, model = {repr(self.model)} relatedPairs = {self.relatedPairs[:3]}... unrelatedPairs = {self.unrelatedPairs[:3]}")
        for pairs, similarities in ((self.relatedPairs, self.relatedPairsSimilarities), (self.unrelatedPairs, self.unrelatedPairsSimilarities)):
            for pair in pairs:
                item1, item2, label = [*pair.values()]
                vec1, vec2 = (self.getVector(item1), self.getVector(item2))
                simScore = self.similarityCheck(vec1, vec2)
                similarities.append(simScore)

        return self.statisticalTest(self.relatedPairsSimilarities, self.unrelatedPairsSimilarities)


class UsersEvaluator(Evaluator):
    def __init__(self, aggregate = np.mean, tokenizer = lambda s: s.split()):
        group1, group0 = EvaluationAdapterFactory.createUsersEvaluationGroups()
        super().__init__(group1, group0, None, limit = 2200)

        self.memorizedUserVectors = {}
        self.aggregate = aggregate
        self.tokenizer = tokenizer

    def getPlotVectors(self):
        vecs = []
        for vec in self.memorizedUserVectors.values():
            vecs.append(vec.cpu().numpy())
            if len(vecs) >= 800:
                break

        return np.array(vecs)

    def getUserVector(self, user):
        vectors = []
        if user["id"] in self.memorizedUserVectors: return self.memorizedUserVectors[user["id"]]

        for proj_id, text in user["projects"].items():
            if proj_id in self.memorizedVectors:
                vectors.append(self.memorizedVectors[proj_id])
            else:
                doc = self.tokenizer(text)
                vectors.append(self.model.call(doc))
                self.memorizedVectors[proj_id] = vectors[-1]

        aggregated = self.aggregate(vectors)
        self.memorizedUserVectors[user["id"]] = aggregated
        return aggregated

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
            return self.getUserVector(obj)
        else:
            # argument is a project
            return self.getProjectVector(obj)

class ProjectsAndUsersEvaluator:
    def __init__(self, combine = np.mean):
        self.projectsEvaluator = None
        self.usersEvaluator = None
        self.combine = combine
        self.logger = None

    def setProjectsEvaluator(self, evaluator):
        self.projectsEvaluator = evaluator

    def setUsersEvaluator(self, evaluator):
        self.usersEvaluator = evaluator

    def setModel(self, model):
        self.projectsEvaluator.setModel(model)
        self.usersEvaluator.setModel(model)

    def setSimilarityCheck(self, fn):
        self.usersEvaluator.setSimilarityCheck(fn)
        self.projectsEvaluator.setSimilarityCheck(fn)

    def evaluate(self):
        self.projectsEvaluator.logger = self.logger
        self.usersEvaluator.logger = self.logger
        projEval = self.projectsEvaluator.evaluate()
        userEval = self.usersEvaluator.evaluate()

        print(f"\nEvaluation values projects: {projEval}, users: {userEval}\n")
        if self.logger != None: self.logger.info(f"\nEvaluation values projects: {projEval}, users: {userEval}\n")
        else: print(f"\nEvaluation values projects: {projEval}, users: {userEval}\n")

        return self.combine([
            projEval,
            userEval
        ])

class Factory:
    @classmethod
    def createDoc2VecEvaluator(cls, model, corpus, similarity):
        evaluator = ProjectsAndUsersEvaluator()
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        projectsEvaluator = Evaluator(relatedAda, unrelatedAda, corpus, limit = 100_000)#135718

        manager = NewDatasetManager(0, inputAdapter = None)
        d2vTokenizer = lambda text: TaggedDocument(words = manager.textPreprocessing(text, False), tags = [0])
        meanAggregator = lambda vectors: np.mean(np.array(vectors), axis = 0)

        usersEvaluator = UsersEvaluator(aggregate = meanAggregator, tokenizer = d2vTokenizer)

        evaluator.setProjectsEvaluator(projectsEvaluator)
        evaluator.setUsersEvaluator(usersEvaluator)
        evaluator.setModel(model)
        evaluator.setSimilarityCheck(similarity)

        return evaluator

    @classmethod
    def createBertEvaluator(cls, model, corpus, similarity):
        def meanAggregator(vectors):
            sum_tensor = torch.sum(torch.stack(vectors), dim=0)

            # Divide by the number of tensors to get the average.
            return sum_tensor / len(vectors)
    
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize(text):
            en = tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=128)

            return en
        
        evaluator = ProjectsAndUsersEvaluator()
        relatedAda, unrelatedAda = EvaluationAdapterFactory.createProjectsEvaluationGroups()
        projectsEvaluator = Evaluator(relatedAda, unrelatedAda, corpus, limit = 100_000)#135718

        usersEvaluator = UsersEvaluator(aggregate = meanAggregator, tokenizer = tokenize)

        evaluator.setProjectsEvaluator(projectsEvaluator)
        evaluator.setUsersEvaluator(usersEvaluator)
        evaluator.setModel(model)
        evaluator.setSimilarityCheck(similarity)

        return evaluator