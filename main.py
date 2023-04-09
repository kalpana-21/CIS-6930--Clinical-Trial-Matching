# !pip install transformers
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess_data import preprocess

# Load data
df = pd.read_csv("Shortened_Data_5000.csv")
df['Summary'].replace('', np.nan, inplace=True)
df.dropna(subset=['Summary'], inplace=True)
df=df.dropna()
# df=df.head(2560)

# Preprocess text
for i in range(len(df)):
#     print(i)
#     if(df.iloc[i,1]==nil)
    df.iloc[i,1] = preprocess(df.iloc[i,1])
    
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

# for i in range(10):
#   print(train_encodings["input_ids"])

# Convert labels to tensors
train_labels = torch.tensor(train_df["Classification"].tolist())
test_labels = torch.tensor(test_df["Classification"].tolist())

batch_size = 16
epochs = 1

# Create PyTorch DataLoader objects
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings["input_ids"]), torch.tensor(train_encodings["attention_mask"]), torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings["input_ids"]), torch.tensor(test_encodings["attention_mask"]), torch.tensor(test_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)
# print("length(Train Loader):", len(train_loader))
# Load BERT model

# Train BERT model
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 5)
model.train()
i=1
for epoch in range(epochs):
    for batch in train_loader:
        print("epoch:",epoch+1,"/",epochs," batch:",i,"/",len(train_loader))
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
    # print("len(batch):",len(batch))
#     print("test_loader",test_loader)
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        print("logits:",logits)
        predictions = torch.argmax(logits, axis=1)
        print("predictions:",predictions)
        print("len(y_true) ",len(y_true)," ")
        y_true += labels.tolist()
        print("len(y_pred) ",len(y_pred))
        y_pred += predictions.tolist()

tp=0
tn=0
fp=0
fn=0

for i in range(len(y_true)):
        if(y_true[i]==y_pred[i]):
            if(y_true[i]==1):
                tp=tp+1
            else:
                tn=tn+1
        else:
            if(y_true[i]==1):
                fn=fn+1
            else:
                fp=fp+1
Accuracy = (tp+tn)/(tp+tn+fp+fn)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F1_Score = (2*Recall*Precision)/(Precision + Recall)

print("Accuracy:", Accuracy)
print("Precision:", Precision)
print("Recall:", Recall)
print("F1 score:", F1_Score)