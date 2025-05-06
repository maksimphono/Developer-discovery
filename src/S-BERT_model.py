import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW


# Example data (replace with your actual data)
data = {
    'document': [
        "Troubleshooting network connectivity issues.",
        "Guide to setting up a VPN connection.",
        "Introduction to Python programming.",
        "Advanced data structures in C++.",
        "Network security best practices.",
        "Python for data analysis.",
        "C++ memory management techniques.",
        "Securing your home network.",
    ],
    'topics': [
        ["networking", "troubleshooting"],
        ["networking", "vpn"],
        ["programming", "python"],
        ["programming", "cpp", "datastructures"],
        ["networking", "security"],
        ["programming", "python", "data analysis"],
        ["programming", "cpp", "memory management"],
        ["networking", "security", "home"],
    ]
}
df = pd.DataFrame(data)

def is_similar(topics1, topics2, threshold=2):
    return len(set(topics1) & set(topics2)) >= threshold

pairs = []
labels = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        doc1 = df['document'][i]
        doc2 = df['document'][j]
        topics1 = df['topics'][i]
        topics2 = df['topics'][j]
        similarity = is_similar(topics1, topics2)
        pairs.append((doc1, doc2))
        labels.append(1 if similarity else 0)

train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SimilarityDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, max_len=128):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        doc1, doc2 = self.pairs[idx]
        label = self.labels[idx]
        encoding1 = self.tokenizer(doc1, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_len)
        encoding2 = self.tokenizer(doc2, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_len)
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(),
            'attention_mask_1': encoding1['attention_mask'].squeeze(),
            'input_ids_2': encoding2['input_ids'].squeeze(),
            'attention_mask_2': encoding2['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

train_dataset = SimilarityDataset(train_pairs, train_labels, tokenizer)
val_dataset = SimilarityDataset(val_pairs, val_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)



class SiameseBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SiameseBert, self).__init__(config)
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.init_weights()

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

model = SiameseBert.from_pretrained('bert-base-uncased')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

def train_epoch(model, dataloader, optimizer, criterion, device):
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

def evaluate_epoch(model, dataloader, criterion, device):
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
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate_epoch(model, val_dataloader, criterion, device)
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