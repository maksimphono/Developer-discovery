import sys
sys.path.append('/home/trukhinmaksim/src')
from time import time

from src.utils.Corpus import MemoryCorpus
from src.utils.CacheAdapter import Factory_21_04_25_HIGH

corp = MemoryCorpus(Factory_21_04_25_HIGH.createFlatAdapter())
#corp.onlyID = True

start = time()
c = 0
for doc in corp:
    c += 1

print(f"Spent : {time() - start}")

print(corp[[2]])