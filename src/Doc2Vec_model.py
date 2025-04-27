import sys
sys.path.append('/home/trukhinmaksim/src')

import numpy as np
import json
import os
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean
from contextlib import redirect_stdout
import logging

from src.utils.CacheAdapter import JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient
from src.utils.Corpus import Corpus
from src.utils.helpers import normalize

import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from annoy import AnnoyIndex


class EXP_CORPUS_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.corpus' object must be an iterable structure, inherited from 'Corpus' class!")

class EXP_MANAGER_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.manager' object must be a DatasetManager instance!")


class AnnoySearcher(AnnoyIndex):
    @classmethod
    def create(cls, vectors, numTrees = 20, distanceType = "angular"):
        obj = cls(vectors[0].size, distanceType)

        normalized = [normalize(v) for v in vectors]

        for i, vec in enumerate(normalized):
            obj.add_item(i, vec)

        obj.build(numTrees)
        return obj


    def selectKmostSimilar(self, vector, k):
        return self.get_nns_by_vector(normalize(vector), k, search_k=-1, include_distances=False)



class Model(gensim.models.doc2vec.Doc2Vec):
    bestParameters = None
    bestScore = 0

    VECTOR_SIZE = 4 #190
    ALPHA_INIT = 0.05
    ALPHA_FINAL = 0.00001

    @classmethod
    def create(cls, trainCorpus, testCorpus, **kwargs):
        model = Model(
                vector_size = cls.VECTOR_SIZE,
                dm_dbow_mode = "DM",
                alpha_init = cls.ALPHA_INIT,
                alpha_final = cls.ALPHA_FINAL,
                **kwargs
            )
        model.trainCorpus = trainCorpus
        model.testCorpus = testCorpus

        return model

    @classmethod
    def configLogger(cls, path):
        logger = logging.getLogger("gensim.models.doc2vec")  # Unique name
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logging.getLogger("gensim.models.doc2vec")

    
    def __init__(self, dm_dbow_mode = "DM", pretrain_w2v = False, alpha_init = 0.05, alpha_final = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainCorpus = None # corpus is an iterator(iterable class object), that will be used in "train" method of Doc2Vec model for data extraction
        self.testCorpus = None # corpuses should be static structures, that are not changing in process of evaluation
        self.alphaInit = alpha_init
        self.alphaFinal = alpha_final
        self.dmDbowMode = dm_dbow_mode
        self.pretrainW2V = pretrain_w2v
        self.logger = logging.getLogger("gensim.models.doc2vec")
        self.normalizedVectors = []
    
    def train(self):
        # will build vocabulary and train the model on trainset (trainset will be fed by corpus)
        
        self.trainCorpus.onlyID(True) # for training, I have to get only id of each vector as tag
        if not isinstance(self.trainCorpus, Corpus): raise EXP_CORPUS_IS_NONE
        #if not isinstance(self.manager, ProjectsDatasetManager): raise EXP_MANAGER_IS_NONE

        start = time()
        self.build_vocab(self.trainCorpus)
        self.logger.info(f"\nVocabulary built in {time() - start} s\n")

        if self.dmDbowMode != "DM+DBOW":
            start = time()
            super().train(
                self.trainCorpus, 
                total_examples = self.corpus_count, 
                epochs = self.epochs,
                start_alpha = self.alphaInit,
                end_alpha = self.alphaFinal
            )
            self.logger.info(f"\nTraining is completed in {time() - start} s\n")
        else:
            # combine DM and DBOW
            pass


    def selectKmostSimilar(self, vector, k):
        simsIndexes = [(0, -np.inf)] # works like monotonic stack, projects with higher score are pushed higher (closer to the end)
        query = vector

        return simsIndexes

    def checkRelevants(self, indexes, tags):
        results = np.zeros(len(indexes))

        for i, doc in enumerate(self.trainCorpus[indexes]):
            if len(set(tags) & set(doc.tags)) > 0: # if that document is actually relevant
                results[i] = 1

        return results

    def test(self, k = 9):
        f1Scores = []
        i = 0

        start = time()
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f): # redirect standard output into void, so 'annoy' doesn't print anything
        
                searcher = AnnoySearcher.create(self.dv.vectors)

                self.trainCorpus.onlyID(False) # for testing all tags are needed
        
                for query in self.testCorpus:
                    vector = self.infer_vector(query.words)
                    topK = sorted(searcher.selectKmostSimilar(vector, k))
                    #topK = [p[0] for p in sorted(self.selectKmostSimilar(vector, k), key = lambda pair: pair[0])]

                    predictedRelevant = np.ones(k)
                    trueRelevant = self.checkRelevants(topK, query.tags)

                    f1Scores.append(f1_score(trueRelevant, predictedRelevant))

                    i += 1

        result = np.mean(f1Scores)
        self.logger.info(f"\nTesting completed in {time() - start} s; Result: {result}")
        return result

    def assess(self, sampleNum = 5, silent = False, format = "full", random_state = None):
        # simple test of model performance
        # take multiple documents from the training corpus and tries to find simillar in the dataset
        # format = "full" | "mean"

        log = lambda s: print(s) if not silent else None
        #performanceGrageScale = {50 : "Random", 60 : "Poor", 70 : "Bad", 80 : "Medium", 92 : "Optimal", 97 : "Perfect"}
        totalDocuments = self.corpus_count
        if random_state != None: randomSeed(random_state)
        indexes = sample(range(totalDocuments), sampleNum)
        if format == "full":
            stats = {}

        i = 0
        avgPerformances = []

        for doc in self.trainCorpus[indexes]:
            vector = self.infer_vector(doc.words)
            sims = defaultdict(lambda: 0, self.dv.most_similar([vector], topn = int(totalDocuments * 0.3)))
            log(f"Assessing document {i} ({doc.tags}). Similarities by tags:")

            if format == "full":
                stats[i] = {
                    "similarities by tags" : {},
                    "average" : 0
                }
            
            for tag in doc.tags:
                if format == "full":
                    stats[i]["similarities by tags"][tag] = sims[tag]
                log(f"  {tag} : {sims[tag]}")
            avgPerformances.append(mean([sims[tag] for tag in doc.tags]))
            log(f"\n  Average similarity value: {avgPerformances[-1]}\n")
            
            if format == "full":
                stats[i]["average"] = avgPerformances[-1]
            i += 1

        if format == "full":
            stats["Average accuracy"] = mean(avgPerformances)
            log(f"Average accuracy: {stats['Average accuracy']}")

            return stats
        else:
            return mean(avgPerformances)

    def evaluate(self): # this method is used be autotuner
        # will train the model on upon-selected set of parameters and test it's performance
        self.train()

        self.trainCorpus.reset()
        result = self.assess(100, silent = True, format = "mean", random_state = 42)

        #self.trainCorpus = CorpusFactory.createFlatTrainDBCorpus_02_04_25_GOOD() # for testing step I must use database adapter for better documents retreival
        #result = self.test(k = 9)

        if Model.bestScore < result:
            Model.bestScore = result
            Model.bestParameters = {
                "min_count" : self.min_count,
                "epochs" : self.epochs,
                "negative" : self.negative,
                "sample" : self.sample,
                "window" : self.window
            }

        return result

    def save(self, fname, *args, **kwargs):
        super().save(fname, *args, **kwargs)
        # Save custom attributes in a separate file
        with open(fname + ".custom_data", 'w') as f:
            for name, attr in self.__getstate__().items():
                print(f"{name} {attr}", file = f)

        self.logger.info(f"Model saved into {fname}")
