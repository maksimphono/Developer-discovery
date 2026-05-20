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

from transformers import AutoModel, BertModel, BertPreTrainedModel, RobertaPreTrainedModel, RobertaModel
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

def areRelevant(tags1, tags2):
    # print("Checking similarity")
    return len(tuple(filter(lambda x: x != 0, set(tags1.tolist()) & set(tags2.tolist())))) >= 1

def createPairsFromBatch(batch):
    pairsBatch = {
        "input_ids_1" : [],
        "input_ids_2" : [],
        "attention_mask_1" : [],
        "attention_mask_2" : [],
        "labels" : []
    }
    for i in range(len(batch['input_ids'])):
        if i >= len(batch['tags']): break
        for j in range(i + 1, len(batch['input_ids'])):
            if j >= len(batch['tags']): break
            # print(len(batch['tags']))
            pairsBatch["input_ids_1"].append(tensor(batch['input_ids'][i]))
            pairsBatch["input_ids_2"].append(tensor(batch['input_ids'][j]))
            pairsBatch["attention_mask_1"].append(tensor(batch['attention_mask'][i]))
            pairsBatch["attention_mask_2"].append(tensor(batch['attention_mask'][j]))

            if areRelevant(batch['tags'][i], batch['tags'][j]):
                pairsBatch["labels"].append(1)
            else:
                pairsBatch["labels"].append(0)

    pairsBatch["input_ids_1"] = torch.stack(pairsBatch["input_ids_1"])
    pairsBatch["input_ids_2"] = torch.stack(pairsBatch["input_ids_2"])
    pairsBatch["attention_mask_1"] = torch.stack(pairsBatch["attention_mask_1"])
    pairsBatch["attention_mask_2"] = torch.stack(pairsBatch["attention_mask_2"])

    pairsBatch["labels"] = torch.tensor(pairsBatch["labels"], dtype=torch.float).unsqueeze(1)

    return pairsBatch

DEFAULT_DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
    def create(cls, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None, name = "bert-base-uncased"):
        model = cls.from_pretrained(name)
        model.prepare(epochs, batchSize, optimizer, device, evaluator)

        return model

    @classmethod
    def load(cls, path, device = DEFAULT_DEVICE):
        model = cls.create()  # Create a new instance of the model
        model.load_state_dict(torch.load(path)) # Load the saved state dictionary
        model.to(device) # Move to the device
        model.eval()

        return model

    def __init__(self, config):
        super(SiameseBert, self).__init__(config)
        self.trainCorpus = None
        self.testCorpus = None
        self.trainDataLoader = None
        self.testDataLoader = None
        self.logger = logging.getLogger(__name__ + '.SiameseBert')
        self.bert = BertModel(config)
        self.output_features = 1 # Define output features
        self.in_features_for_manual = 1
        self.W = nn.Parameter(torch.randn(self.output_features, self.in_features_for_manual))  # Use nn.Parameter
        self.b = nn.Parameter(torch.randn(self.output_features))
        self.init_weights()

        self.logger.info(f"Bert model initialized")

    def prepare(self, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        self.batchSize = batchSize
        self.epochs = epochs
        self.evaluator = evaluator
        self.dev = device
        self.criterion = nn.BCEWithLogitsLoss() # most suitable criterion for Siamise BERT

        if optimizer == None:
            self.optimizer = AdamW(self.parameters(), lr=1e-5)
        else:
            self.optimizer = optimizer

        self.logger.info(f"Bert model prepared with epochs = {epochs}, batch size = {batchSize}, optimizer = {self.optimizer}, device = {self.dev}")
        # print("Bert prepared")

    def setTrainCorpus(self, corpus):
        self.trainCorpus = corpus
        self.trainDataLoader = DataLoader(self.trainCorpus, batch_size=self.batchSize, shuffle=True)
        self.logger.info(f"Train DataLoader is set, length = {len(self.trainCorpus)}")
        # print("self.trainDataLoader is set")

    def setTestCorpus(self, corpus):
        self.testCorpus = corpus
        self.testDataLoader = DataLoader(self.testCorpus, batch_size=self.batchSize)
        self.logger.info(f"Test DataLoader is set, length = {len(self.testCorpus)}")
        # print("self.testDataLoader is set")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Get the embeddings of the [CLS] token (representing the whole sequence)
        embedding1 = output1.pooler_output
        embedding2 = output2.pooler_output

        # Calculate similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1).unsqueeze(1)

        prediction = torch.bmm(
            similarity.view(
                similarity.size(0), 
                1, 
                self.in_features_for_manual
            ), 
            self.W.unsqueeze(0).expand(
                similarity.size(0), 
                self.output_features, 
                self.in_features_for_manual
            ).transpose(1, 2)
        ) + self.b.unsqueeze(0)
        prediction = prediction.squeeze(1)
        return prediction

    def unpackBatch(self, batch):
        pairs = createPairsFromBatch(batch)
        return (
            pairs['input_ids_1'].to(self.dev),
            pairs['attention_mask_1'].to(self.dev),
            pairs['input_ids_2'].to(self.dev),
            pairs['attention_mask_2'].to(self.dev),
            pairs['labels'].to(self.dev).unsqueeze(1)
        )

    def trainEpoch(self):
        self.to(self.dev)
        self.train()
        totalLoss = 0

        for batch in self.trainDataLoader:
            # unpacking one batch
            input_ids_1 = batch['input_ids_1'].to(self.dev)
            attention_mask_1 = batch['attention_mask_1'].to(self.dev)
            input_ids_2 = batch['input_ids_2'].to(self.dev)
            attention_mask_2 = batch['attention_mask_2'].to(self.dev)
            labels = batch['labels'].to(self.dev).unsqueeze(1)

            # training one batch
            self.optimizer.zero_grad()
            outputs = self(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            #print("Labels shape:", labels.shape)
            #print("Inputs shape:", outputs.shape)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item()

            # print("Train epoch completed")

        return totalLoss / len(self.trainDataLoader)

    def evalEpoch(self):
        self.eval()
        totalLoss = 0
        correctPredictions = 0
        totalPairsNum = 0

        with torch.no_grad():
            for batch in self.testDataLoader:
                #pairs = createPairsFromBatch(batch)
                input_ids_1 = batch['input_ids_1'].to(self.dev)
                attention_mask_1 = batch['attention_mask_1'].to(self.dev)
                input_ids_2 = batch['input_ids_2'].to(self.dev)
                attention_mask_2 = batch['attention_mask_2'].to(self.dev)
                labels = batch['labels'].to(self.dev).unsqueeze(1)
                # print("batch unpacked")

                outputs = self(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.criterion(outputs, labels)
                totalLoss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                correctPredictions += (predictions == labels).sum().item()
                totalPairsNum += labels.size(0)

                # print("Eval epoch completed")

        meanLoss = totalLoss / len(self.testDataLoader)
        accuracy = correctPredictions / totalPairsNum

        return meanLoss, accuracy

    def trainMe(self):
        start = time()

        self.logger.info(f"Training started, epochs = {self.epochs}\n")
        for epoch in range(self.epochs):
            trainLoss = self.trainEpoch()
            evalLoss, evalAccuracy = self.evalEpoch()

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")
            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")

        self.logger.info(f"\nTraining is completed in {time() - start}")
        # print(f"\nTraining is completed in {time() - start}")

    def call(self, document):
        self.eval()
        # print(f"model is called with {document['input_ids']}")
        #encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)  # Or your desired max_length
        input_ids = document['input_ids'].to(self.dev) #.unsqueeze(0)
        attention_mask = document['attention_mask'].to(self.dev)

        with torch.no_grad():  # Ensure no gradients are calculated during inference
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return output.squeeze(0)#.to(torch.device("cpu"))

    def evaluate(self, beforeEvaluation = lambda: None):
        start = 0
        result = 0
        self.eval()

        self.trainCorpus.reset()

        if self.evaluator != None:
            beforeEvaluation()
            start = time()
            self.evaluator.setModel(self)
            self.evaluator.logger = self.logger
            result = self.evaluator.evaluate()

        self.logger.info(f"\nEvaluation is completed in {time() - start}; Result = {result}\n")
        # print(f"\nEvaluation is completed in {time() - start}; Result = {result}\n")

        return result

    def save(self, path):
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")


class SiameseRoBerta(RobertaPreTrainedModel):
    @classmethod
    def configLogger(cls, path):
        logger = logging.getLogger(__name__ + '.SiameseRoBerta')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @classmethod
    def create(cls, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        model = cls.from_pretrained("roberta-base")
        model.prepare(epochs, batchSize, optimizer, device, evaluator)

        return model

    @classmethod
    def load(cls, path, device = DEFAULT_DEVICE):
        model = cls.create()  # Create a new instance of the model
        model.load_state_dict(torch.load(path)) # Load the saved state dictionary
        model.to(device) # Move to the device
        model.eval()

        return model

    def __init__(self, config):
        super(SiameseRoBerta, self).__init__(config)
        self.trainCorpus = None
        self.testCorpus = None
        self.trainDataLoader = None
        self.testDataLoader = None
        self.logger = logging.getLogger(__name__ + '.SiameseRoBerta')
        self.bert = RobertaModel(config)
        self.output_features = 1 # Define output features
        self.in_features_for_manual = 1
        self.W = nn.Parameter(torch.randn(self.output_features, self.in_features_for_manual))  # Use nn.Parameter
        self.b = nn.Parameter(torch.randn(self.output_features))
        #self.fc = nn.Linear(config.hidden_size, 1)
        self.init_weights()

        self.logger.info(f"RoBerta model initialized")
        # print("Bert initialized")

    def prepare(self, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        self.batchSize = batchSize
        self.epochs = epochs
        self.evaluator = evaluator
        self.dev = device
        self.criterion = nn.BCEWithLogitsLoss() # most suitable criterion for Siamise BERT

        if optimizer == None:
            self.optimizer = AdamW(self.parameters(), lr=1e-6)
        else:
            self.optimizer = optimizer

        self.logger.info(f"RoBerta model prepared with epochs = {epochs}, batch size = {batchSize}, optimizer = {self.optimizer}, device = {self.dev}")
        # print("Bert prepared")

    def setTrainCorpus(self, corpus):
        self.trainCorpus = corpus
        self.trainDataLoader = DataLoader(self.trainCorpus, batch_size=self.batchSize, shuffle=True)
        self.logger.info(f"Train DataLoader is set, length = {len(self.trainCorpus)}")
        # print("self.trainDataLoader is set")

    def setTestCorpus(self, corpus):
        self.testCorpus = corpus
        self.testDataLoader = DataLoader(self.testCorpus, batch_size=self.batchSize)
        self.logger.info(f"Test DataLoader is set, length = {len(self.testCorpus)}")
        # print("self.testDataLoader is set")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Get the embeddings of the [CLS] token (representing the whole sequence)
        embedding1 = output1.last_hidden_state[:, 0, :]
        embedding2 = output2.last_hidden_state[:, 0, :]

        # Calculate similarity (e.g., cosine similarity followed by a linear layer)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1).unsqueeze(1)
        #prediction = self.fc(similarity.unsqueeze(1))
        prediction = torch.bmm(
            similarity.view(
                similarity.size(0), 
                1, 
                self.in_features_for_manual
            ), 
            self.W.unsqueeze(0).expand(
                similarity.size(0), 
                self.output_features, 
                self.in_features_for_manual
            ).transpose(1, 2)
        ) + self.b.unsqueeze(0)
        prediction = prediction.squeeze(1)

        return prediction

    def unpackBatch(self, batch):
        pairs = createPairsFromBatch(batch)
        return (
            pairs['input_ids_1'].to(self.dev),
            pairs['attention_mask_1'].to(self.dev),
            pairs['input_ids_2'].to(self.dev),
            pairs['attention_mask_2'].to(self.dev),
            pairs['labels'].to(self.dev).unsqueeze(1)
        )

    def trainEpoch(self):
        self.to(self.dev)
        self.train()
        totalLoss = 0

        for batch in self.trainDataLoader:
            # unpacking one batch
            input_ids_1 = batch['input_ids_1'].to(self.dev)
            attention_mask_1 = batch['attention_mask_1'].to(self.dev)
            input_ids_2 = batch['input_ids_2'].to(self.dev)
            attention_mask_2 = batch['attention_mask_2'].to(self.dev)
            labels = batch['labels'].to(self.dev).unsqueeze(1)

            # training one batch
            self.optimizer.zero_grad()
            outputs = self(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item()

        return totalLoss / len(self.trainDataLoader)

    def evalEpoch(self):
        self.eval()
        totalLoss = 0
        correctPredictions = 0
        totalPairsNum = 0

        with torch.no_grad():
            for batch in self.testDataLoader:
                #pairs = createPairsFromBatch(batch)
                # unpacking one batch
                input_ids_1 = batch['input_ids_1'].to(self.dev)
                attention_mask_1 = batch['attention_mask_1'].to(self.dev)
                input_ids_2 = batch['input_ids_2'].to(self.dev)
                attention_mask_2 = batch['attention_mask_2'].to(self.dev)
                labels = batch['labels'].to(self.dev).unsqueeze(1)

                # training one batch
                outputs = self(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.criterion(outputs, labels)
                totalLoss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.45

                correctPredictions += (predictions == labels).sum().item()
                totalPairsNum += labels.size(0)

        meanLoss = totalLoss / len(self.testDataLoader)
        accuracy = correctPredictions / totalPairsNum

        return meanLoss, accuracy

    def trainMe(self):
        start = time()

        self.logger.info(f"Training started, epochs = {self.epochs}\n")
        for epoch in range(self.epochs):
            trainLoss = self.trainEpoch()
            evalLoss, evalAccuracy = self.evalEpoch()

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")
            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")

        self.logger.info(f"\nTraining is completed in {time() - start}")
        # print(f"\nTraining is completed in {time() - start}")

    def call(self, document):
        self.eval()
        # print(f"model is called with {document['input_ids']}")
        input_ids = tensor(document['input_ids']).unsqueeze(0).to(self.dev)
        attention_mask = tensor(document['attention_mask']).unsqueeze(0).to(self.dev)

        with torch.no_grad():  # Ensure no gradients are calculated during inference
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return output

    def evaluate(self, beforeEvaluation = lambda: None):
        start = 0
        result = 0
        self.eval()

        self.trainCorpus.reset()

        if self.evaluator != None:
            beforeEvaluation()
            start = time()
            self.evaluator.setModel(self)
            self.evaluator.logger = self.logger
            result = self.evaluator.evaluate()

        self.logger.info(f"\nEvaluation is completed in {time() - start}; Result = {result}\n")

        return result

    def save(self, path):
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
