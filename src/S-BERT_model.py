import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

#train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

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
            pairsBatch["input_ids_1"].append(doc1['input_ids'].squeeze())
            pairsBatch["input_ids_2"].append(doc2['input_ids'].squeeze())
            pairsBatch["attention_mask_1"].append(doc1['attention_mask'].squeeze())
            pairsBatch["attention_mask_2"].append(doc2['attention_mask'].squeeze())

            if areRelevant(doc1, doc2):
                pairsBatch["labels"].append(torch.tensor(1, dtype=torch.float))
            else:
                pairsBatch["labels"].append(torch.tensor(0, dtype=torch.float))

    pairsBatch["input_ids_1"] = tensor(pairsBatch["input_ids_1"])
    pairsBatch["input_ids_2"] = tensor(pairsBatch["input_ids_2"])
    pairsBatch["attention_mask_1"] = tensor(pairsBatch["attention_mask_1"])
    pairsBatch["attention_mask_2"] = tensor(pairsBatch["attention_mask_2"])
    pairsBatch["labels"] = tensor(pairsBatch["labels"])

    return pairsBatch

trainCorpus = SimilarityDataset(train_pairs, train_labels, tokenizer)
val_dataset = SimilarityDataset(val_pairs, val_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseBert(BertPreTrainedModel):
    def __init__(self, config, epochs = 5, batchSize = 16, optimizer = None, device = DEFAULT_DEVICE, evaluator = None):
        super(SiameseBert, self).__init__(config)
        self.trainCorpus = None
        self.testCorpus = None
        self.logger = None
        self.batchSize = batchSize
        self.epochs = epochs
        self.evaluator = evaluator
        self.trainDataLoader = None
        self.testDataLoader = None
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss() # most suitable criterion for Siamise BERT
        if optimizer == None:
            self.optimizer = AdamW(self.parameters(), lr=1e-5)
        else:
            self.optimizer = optimizer

        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def setTrainCorpus(self, corpus):
        self.trainCorpus = corpus
        self.trainDataLoader = DataLoader(self.trainCorpus, batch_size=self.batchSize, shuffle=True)

    def setTestCorpus(self, corpus):
        self.testCorpus = corpus
        self.testDataLoader = DataLoader(self.testCorpus, batch_size=self.batchSize)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Get the embeddings of the [CLS] token (representing the whole sequence)
        embedding1 = output1.pooler_output
        embedding2 = output2.pooler_output

        # Calculate similarity (e.g., cosine similarity followed by a linear layer)
        similarity = F.cosine_similarity(embedding1, embedding2)
        prediction = self.fc(similarity.unsqueeze(1))
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
        total_loss = 0

        for batch in self.trainDataLoader:
            pairs = createPairsFromBatch(batch)
            input_ids_1 = pairs['input_ids_1'].to(self.device)
            attention_mask_1 = pairs['attention_mask_1'].to(self.device)
            input_ids_2 = pairs['input_ids_2'].to(self.device)
            attention_mask_2 = pairs['attention_mask_2'].to(self.device)
            labels = pairs['labels'].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = super().__call__(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.trainDataLoader)

    def evalEpoch(self):
        self.eval()
        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in self.testDataLoader:
                pairs = createPairsFromBatch(batch)
                input_ids_1 = pairs['input_ids_1'].to(self.device)
                attention_mask_1 = pairs['attention_mask_1'].to(self.device)
                input_ids_2 = pairs['input_ids_2'].to(self.device)
                attention_mask_2 = pairs['attention_mask_2'].to(self.device)
                labels = pairs['labels'].to(self.device).unsqueeze(1)

                outputs = super().__call__(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                correct_predictions += (predictions == labels).sum().item()

        avg_loss = total_loss / len(self.testDataLoader)
        accuracy = correct_predictions / len(self.testCorpus)

        return avg_loss, accuracy

    def train(self):
        for epoch in range(self.epochs):
            trainLoss = trainEpoch()
            evalLoss, evalAccuracy = evalEpoch()

            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {trainLoss:.4f}, Val Loss: {evalLoss:.4f}, Val Accuracy: {evalAccuracy:.4f}")

    def __call__(self, document):
        model.eval()

        with torch.no_grad():
            outputs = model.bert(**document).pooler_output
        return outputs.cpu().numpy()
        #return super().__call__(*args, **kwargs)

    def evaluate(self):
        pass


model = SiameseBert.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

def trainEpoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evalEpoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

            outputs = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predictions == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = trainEpoch(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_accuracy = evalEpoch(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# To get document embeddings after training:
def get_document_embedding(text, model, tokenizer, device, max_len=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len).to(device)
    with torch.no_grad():
        outputs = model.bert(**encoding).pooler_output
    return outputs.cpu().numpy()

sample_doc = "Troubleshooting network connectivity issues."
embedding = get_document_embedding(sample_doc, model, tokenizer, device)
print(f"Embedding shape: {embedding.shape}")