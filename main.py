# !pip install transformers
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess_data import preprocess

# Load data
df = pd.read_csv("Shortened_Data.csv")
df=df.head(100)

# Preprocess text
for i in range(len(df)):
  text = preprocess(df.iloc[i,1])

# Split data into train/test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#  Clinical Bio Bert 
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#   Bio Bert 
# model = AutoModelForSequenceClassification.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")
# tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")

# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# Tokenize text
train_encodings = tokenizer(train_df["Summary"].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df["Summary"].tolist(), truncation=True, padding=True, max_length=512)

for i in range(10):
  print(train_encodings["input_ids"])

# Convert labels to tensors
train_labels = torch.tensor(train_df["Classification"].tolist())
test_labels = torch.tensor(test_df["Classification"].tolist())

# Create PyTorch DataLoader objects
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings["input_ids"]), torch.tensor(train_encodings["attention_mask"]), torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings["input_ids"]), torch.tensor(test_encodings["attention_mask"]), torch.tensor(test_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load BERT model

# Train BERT model
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 5)
model.train()
i=1
for epoch in range(1):
    for batch in train_loader:
        print("batch: ",i,"\n")
        i=i+1
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluate BERT model
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1)
        y_true += labels.tolist()
        y_pred += predictions.tolist()

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred,average='micro'))
print("Recall:", recall_score(y_true, y_pred,average='micro'))
print("F1 score:", f1_score(y_true, y_pred,average='micro'))