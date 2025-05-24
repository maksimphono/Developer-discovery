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

from src.utils.DatasetManager import ReadmeFilesTranslatonManager
from src.utils.CacheAdapter import FlatAdapter

N = 0

class Out:
    def save(self, items):
        #print(items)
        print([i["proj_id"] for i in items])

inputAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/raw_readme_30-04-25")
outputAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_30-04-25/translated_readme_30-04-25")
manager = ReadmeFilesTranslatonManager(itemsPortionNum = 10, maxLength = 4000, limit = 48306, skip = N * 48306, inputAdapter=inputAdapter, outputAdapters = [outputAdapter])

start = time()
asyncio.run(manager.call())

print(f"Spent: {time() - start}")