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

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

def areRelevant(doc1, doc2):
    return len(set(doc1["tags"]) & set(doc2["tags"])) >= 1

def createPairsFromBatch(batch):
    pairsBatch = {
        "input_ids_1" : [],
        "input_ids_2" : [],
        "attention_mask_1" : [],
        "attention_mask_2" : [],
        "labels" : []
    }
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            doc1 = batch[i]
            doc2 = batch[j]
            pairsBatch["input_ids_1"].append(tensor(doc1['input_ids']))
            pairsBatch["input_ids_2"].append(tensor(doc2['input_ids']))
            pairsBatch["attention_mask_1"].append(tensor(doc1['attention_mask']))
            pairsBatch["attention_mask_2"].append(tensor(doc2['attention_mask']))

            if areRelevant(doc1, doc2):
                pairsBatch["labels"].append(1)
            else:
                pairsBatch["labels"].append(0)

    pairsBatch["input_ids_1"] = torch.stack(pairsBatch["input_ids_1"])
    pairsBatch["input_ids_2"] = torch.stack(pairsBatch["input_ids_2"])
    pairsBatch["attention_mask_1"] = torch.stack(pairsBatch["attention_mask_1"])
    pairsBatch["attention_mask_2"] = torch.stack(pairsBatch["attention_mask_2"])

    pairsBatch["labels"] = torch.tensor(pairsBatch["labels"], dtype=torch.float).unsqueeze(1)

    return pairsBatch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseBert(BertPreTrainedModel):
    @classmethod
    def configLogger(cls, path):
        logger = logging.getLogger(__name__ + '.SiameseBert')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @classmethod
    def create(cls, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        model = cls.from_pretrained('bert-base-uncased')
        model.prepare(epoch, batchSize, optimizer, device, evaluator)

        return model

    @classmethod
    def load(cls, path, device = DEFAULT_DEVICE):
        model = SiameseBert('bert-base-uncased')  # Create a new instance of the model
        model.load_state_dict(torch.load(path)) # Load the saved state dictionary
        model.to(device) # Move to the device
        model.eval()

        return loaded_model

    def __init__(self, config):
        super(SiameseBert, self).__init__(config)
        self.trainCorpus = None
        self.testCorpus = None
        self.trainDataLoader = None
        self.testDataLoader = None
        self.logger = logging.getLogger(__name__ + '.SiameseBert')
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.init_weights()

        print("Bert initialized")

    def prepare(self, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        self.batchSize = batchSize
        self.epochs = epochs
        self.evaluator = evaluator
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss() # most suitable criterion for Siamise BERT
        if optimizer == None:
            self.optimizer = AdamW(self.parameters(), lr=1e-5)
        else:
            self.optimizer = optimizer

        print("Bert prepared")

    def setTrainCorpus(self, corpus):
        self.trainCorpus = corpus
        self.trainDataLoader = DataLoader(self.trainCorpus, batch_size=self.batchSize, shuffle=True)
        print("self.trainDataLoader is set")

    def setTestCorpus(self, corpus):
        self.testCorpus = corpus
        self.testDataLoader = DataLoader(self.testCorpus, batch_size=self.batchSize)
        print("self.testDataLoader is set")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Get the embeddings of the [CLS] token (representing the whole sequence)
        embedding1 = output1.pooler_output
        embedding2 = output2.pooler_output

        # Calculate similarity (e.g., cosine similarity followed by a linear layer)
        similarity = F.cosine_similarity(embedding1, embedding2)
        prediction = self.fc(similarity.unsqueeze(1))
        print("forward is called")
        return prediction

    def unpackBatch(self, batch):
        pairs = createPairsFromBatch(rawBatch)
        return (
            pairs['input_ids_1'].to(self.device)
            pairs['attention_mask_1'].to(self.device)
            pairs['input_ids_2'].to(self.device)
            pairs['attention_mask_2'].to(self.device)
            pairs['labels'].to(self.device).unsqueeze(1)
        )

    def trainEpoch(self):
        self.to(self.device)
        self.train()
        totalLoss = 0

        for batch in self.trainDataLoader:
            pairs = createPairsFromBatch(batch)
            input_ids_1 = pairs['input_ids_1'].to(self.device)
            attention_mask_1 = pairs['attention_mask_1'].to(self.device)
            input_ids_2 = pairs['input_ids_2'].to(self.device)
            attention_mask_2 = pairs['attention_mask_2'].to(self.device)
            labels = pairs['labels'].to(self.device)

            print("batch unpacked")
            self.optimizer.zero_grad()
            outputs = super().__call__(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item()

            print("Train epoch completed")

        return totalLoss / len(self.trainDataLoader)

    def evalEpoch(self):
        self.eval()
        totalLoss = 0
        correctPredictions = 0
        totalPairsNum = 0

        with torch.no_grad():
            for batch in self.testDataLoader:
                pairs = createPairsFromBatch(batch)
                input_ids_1 = pairs['input_ids_1'].to(self.device)
                attention_mask_1 = pairs['attention_mask_1'].to(self.device)
                input_ids_2 = pairs['input_ids_2'].to(self.device)
                attention_mask_2 = pairs['attention_mask_2'].to(self.device)
                labels = pairs['labels'].to(self.device)
                print("batch unpacked")

                outputs = super().__call__(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.criterion(outputs, labels)
                totalLoss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                correctPredictions += (predictions == labels).sum().item()
                totalPairsNum += labels.size(0)

                print("Eval epoch completed")

        meanLoss = totalLoss / len(self.testDataLoader)
        accuracy = correctPredictions / totalPairsNum

        return meanLoss, accuracy

    def train(self):
        start = time()
        for epoch in range(self.epochs):
            trainLoss = self.trainEpoch()
            evalLoss, evalAccuracy = self.evalEpoch()

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")

        self.logger.info(f"\nTraining is completed in {time() - start}")

    def __call__(self, document):
        model.eval()
        #encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)  # Or your desired max_length
        input_ids = tensor(document['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = tensor(document['attention_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():  # Ensure no gradients are calculated during inference
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return output

    def evaluate(self):
        start = 0
        result = 0
        self.train()

        self.trainCorpus.reset()

        if self.evaluator != None:
            start = time()
            self.evaluator.setModel(self)
            result = self.evaluator.evaluate()

        self.logger.info(f"\nEvaluation is completed in {time() - start}; Result = {result}\n")

        return result

    def save(self, path):
        torch.save(self.state_dict(), path)
        self.logger(f"Model saved to {path}")
