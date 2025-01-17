import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pickle

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, 
                                 max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels.tolist())

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class BertNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def train_model(model, train_loader, val_loader, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        print(f'Epoch {epoch+1} Loss: {total_loss/len(train_loader)}')
        print(classification_report(val_labels, val_preds))

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10,7))
    sb.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load data
    train_df = pd.read_csv('dataset/train.csv')
    new_data = pd.read_csv('new_articles.csv')
    
    # Preprocessing
    train_df = train_df.dropna()
    combined_data = pd.concat([train_df['text'], new_data['content']], ignore_index=True)
    combined_labels = pd.concat([train_df['label'], new_data['label']], ignore_index=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, combined_labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertNewsClassifier()
    
    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Train model
    train_model(model, train_loader, test_loader)
    
    # Save model
    torch.save(model.state_dict(), 'bert_model.pth')
    
    # Evaluation on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            test_preds.extend(predictions.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    # Print metrics
    print("\nTest Set Performance:")
    print(classification_report(test_labels, test_preds))
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, ['REAL', 'FAKE'])

if __name__ == "__main__":
    main()