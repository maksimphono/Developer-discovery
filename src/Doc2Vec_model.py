import sys
sys.path.append('/home/trukhinmaksim/src')

import numpy as np
import json
from time import time
from random import sample, seed as randomSeed
from collections import defaultdict
from numpy import mean
import logging

from src.utils.CacheAdapter import JSONAdapter, JSONMultiFileAdapter, EXP_END_OF_DATA
from src.utils.DatasetManager import ProjectsDatasetManager
from src.utils.validators import projectDataIsSufficient

import gensim

from src.utils.Corpus import Corpus

class EXP_CORPUS_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.corpus' object must be an iterable structure, inherited from 'Corpus' class!")

class EXP_MANAGER_IS_NONE(Exception):
    def __init__(self):
        super().__init__("'Model.manager' object must be a DatasetManager instance!")


class Model(gensim.models.doc2vec.Doc2Vec):
    manager = None
    corpus = None
    bestParameters = None
    bestScore = 0

    VECTOR_SIZE = 4 #190
    ALPHA_INIT = 0.05
    ALPHA_FINAL = 0.00001

    @classmethod
    def create(cls, **kwargs):
        model = Model(
                vector_size = cls.VECTOR_SIZE,
                dm_dbow_mode = "DM", 
                alpha_init = cls.ALPHA_INIT,
                alpha_final = cls.ALPHA_FINAL,
                **kwargs
            )
        cls.manager.cacheAdapter.reset()
        cls.manager.clearData()
        model.corpus = cls.corpus

        return model
    
    def __init__(self, dm_dbow_mode = "DM", pretrain_w2v = False, alpha_init = 0.05, alpha_final = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainCorpus = None # corpus is an iterator(iterable class object), that will be used in "train" method of Doc2Vec model for data extraction
        self.testCorpus = None # corpuses should be static structures, that are not changing in process of evaluation
        self.relevant = []
        self.alphaInit = alpha_init
        self.alphaFinal = alpha_final
        self.dmDbowMode = dm_dbow_mode
        self.pretrainW2V = pretrain_w2v
    
    def train(self):
        # will build vocabulary and train the model on corpus (corpus will be fed by corpus)
        
        if not isinstance(self.trainCorpus, Corpus): raise EXP_CORPUS_IS_NONE
        #if not isinstance(self.manager, ProjectsDatasetManager): raise EXP_MANAGER_IS_NONE

        start = time()
        self.build_vocab(self.trainCorpus)
        logging.info(f"\nVocabulary built in {time() - start} s\n")

        if self.dmDbowMode != "DM+DBOW":
            start = time()
            super().train(
                self.trainCorpus, 
                total_examples = self.trainCorpus_count, 
                epochs = self.epochs,
                start_alpha = self.alphaInit,
                end_alpha = self.alphaFinal
            )
            logging.info(f"\nTraining is completed in {time() - start} s\n")
        else:
            # combine DM and DBOW
            pass

    def constructRelevant(self):
        # note, that items are placed in the list in the same order as in test corpus, so order of items in corpus shouldn't change
        # TODO: complete this method for relevant collection
        """
        # pseudocode:
        for query in test_set:
            for doc in train_set:
                # find all vectors in my train set, that are relevant to the query
                if share_common_tags(doc, query):
                    query.add_relevant(doc)
        """
        for query in self.testCorpus:
            for doc in self.trainCorpus:
                pass

    def test(self):
        # TODO: complete this method with the new evaluation technique
        """
        # preudocode:
        f1_scores = []
        for query in test set:
            vec = model.get_vector(query)
            topK = vec.selct_K_most_similar()
            p = calculate_precision(topK, query.get_relevants())
            r = calculate_recall(topK, query.get_relevants())
            f1_scores.add( calculate_f1(p, r) )

        return mean(f1_scores)
        """
        pass

    def assess(self, sampleNum = 5, silent = False, format = "full", random_state = None):
        # simple test of model performance
        # take multiple documents from the training corpus and tries to find simillar in the dataset
        # format = "full" | "mean"

        log = lambda s: print(s) if not silent else None
        #performanceGrageScale = {50 : "Random", 60 : "Poor", 70 : "Bad", 80 : "Medium", 92 : "Optimal", 97 : "Perfect"}
        totalDocuments = self.trainCorpus_count
        if random_state != None: randomSeed(random_state)
        indexes = sample(range(totalDocuments), sampleNum)
        if format == "full":
            stats = {}

        i = 0
        avgPerformances = []

        for doc in self.trainCorpus:
            if i >= totalDocuments: break
            if i in indexes:
                vector = self.infer_vector(doc.words)
                sims = defaultdict(lambda: 0, self.dv.most_similar([vector], topn = totalDocuments))

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

    def evaluate(self):
        # will train the model on upon-selected set of parameters and test it's performance
        self.train()

        result = self.assess(6, silent = True, format = "mean", random_state = 42)

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
