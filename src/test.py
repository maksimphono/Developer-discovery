import sys
sys.path.append('/home/trukhinmaksim/src')
from time import time

from src.utils.Corpus import MemoryCorpus, Factory_21_04_25_HIGH as CorpusFactory
from src.utils.CacheAdapter import Factory_21_04_25_HIGH
from torch.utils.data import DataLoader

testCorp = CorpusFactory.BERT.createTestPairsCorpus(16)

loader = DataLoader(testCorp, batch_size=4, shuffle=False)

for i, batch in enumerate(loader):
    #print(batch["input_ids_1"][0])
    #print(batch["input_ids_2"][0])
    #print(batch["labels"].unsqueeze(1).shape)

    if i >= 1: break