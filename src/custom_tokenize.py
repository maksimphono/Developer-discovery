import sys
sys.path.append('/home/trukhinmaksim/src')
import os
from random import sample, random
from time import sleep
from json import dumps

#from pymongo.errors import ServerSelectionTimeoutError, CursorNotFound

from src.utils.DatasetManager import DatasetManager, NewDatasetManager
from src.utils.CacheAdapter import CacheAdapter, EXP_END_OF_DATA, FlatAdapter, Factory_21_04_25_HIGH as CacheFactory
from src.utils.validators import projectDataIsHighQuality

from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


TOKENIZED_SAVE_PATH = "/home/trukhinmaksim/src/data/cache_21-04-25"
MAX_LEN = 128

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(doc):
    encoding = tokenizer(doc["text"], truncation=True, padding='max_length', max_length=MAX_LEN)
    encoding["tags"] = [hash(tag) for tag in doc["tags"]] + [0] * (8 - len(doc["tags"]))

    return dict(encoding)

def main():
    textAdapter = CacheFactory.createTextTestAdapter()
    tokenizedAdapter = FlatAdapter("/home/trukhinmaksim/src/data/cache_21-04-25/test_tokenized_BERT_21-04-25")

    manager = DatasetManager(
        5000,
        inputAdapter = textAdapter,
        outputAdapters = [tokenizedAdapter],
        mapper = tokenize
    )

    i = 0
    while 1:
        try:
            manager()
            print(f"Tokeinzed {i} documents")
            i += 5000
        except EXP_END_OF_DATA:
            break

    c = 0
    def count(doc):
        nonlocal c
        c += 1
        return None

    manager = DatasetManager(
        10000,
        inputAdapter = tokenizedAdapter,
        outputAdapters = [],
        mapper = count
    )
    while 1:
        try:
            manager()
        except EXP_END_OF_DATA:
            break
    print(f"Len = {c}")

if __name__ == "__main__":
    main()