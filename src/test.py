import sys
sys.path.append('/home/trukhinmaksim/src')
import json
from time import time
from collections import defaultdict
from numpy import array
from random import random, sample, shuffle
import numpy as np
from copy import deepcopy
import asyncio
from src.utils.Corpus import MemoryCorpus
from src.utils.CacheAdapter import Factory_21_04_25_HIGH

from src.utils.DatasetManager import ReadmeFilesTranslatonManager
from src.utils.CacheAdapter import FlatAdapter

class Out:
    def save(self, items):
        #print(items)
        print([i["proj_id"] for i in items])

inputAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
output = Out()
manager = ReadmeFilesTranslatonManager(itemsPortionNum = 4, maxLength = 2000, limit = 10, inputAdapter=inputAdapter, outputAdapters = [output])

start = time()
asyncio.run(manager.call())

print(f"Spent: {time() - start}")