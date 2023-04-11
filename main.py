# !pip install transformers
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

#Preprocess the text - removal of punctuations, casing and stopwords
def preprocess(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in stop_words]
    text = ' '.join(tokens)
    return text

#Data size taken to train and test the model
data_size = 20000

#No. of batches and epochs
batch_size = 16
epochs = 1

#We split the dataset into 1-labelled and 0-labelled to train the model with equal distribution of both labels
# Load the 1-labelled dataset
df_1label = pd.read_csv("Data Set_1_label.csv")
df_1label=df_1label.head(int(data_size/2))

# Load the 1-labelled dataset
df_0label = pd.read_csv("Data_Set_0_label.csv")
df_0label=df_0label.head(int(data_size/2))

#concat both subsets to provide to the model for training
df = pd.concat([df_1label, df_0label], axis=0)
df['Summary'].replace('', np.nan, inplace=True)

#drop any empty rows
df.dropna(subset=['Summary'], inplace=True)
df = df.dropna()
df = df.sample(frac = 1)

# Preprocess text
for i in range(len(df)):
    df.iloc[i,1] = preprocess(df.iloc[i,1])
    
# Split into test and train
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#  Clinical Bio Bert 
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#   Bio Bert 
# model = AutoModelForSequenceClassification.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")
# tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")

# Tokenize text
train_encodings = tokenizer(train_df["Summary"].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df["Summary"].tolist(), truncation=True, padding=True, max_length=512)

# Convert labels to tensors
train_labels = torch.tensor(train_df["Classification"].tolist())
test_labels = torch.tensor(test_df["Classification"].tolist())


# Create PyTorch DataLoader objects
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings["input_ids"]), torch.tensor(train_encodings["attention_mask"]), torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings["input_ids"]), torch.tensor(test_encodings["attention_mask"]), torch.tensor(test_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

# Train the BERT model
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
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1)
        y_true += labels.tolist()
        y_pred += predictions.tolist()

#Metrics to compute f1 score
#true positives, true negatives, false positives, false negatives
tp=0
tn=0
fp=0
fn=0

result=[]
for i in range(len(y_true)):
        if(y_true[i]==y_pred[i]):
            result.append("Met")
            if(y_true[i]==1):
                tp=tp+1
            else:
                tn=tn+1
        else:
            result.append("Not Met")
            if(y_true[i]==1):
                fn=fn+1
            else:
                fp=fp+1
                
test_df['Classification Result'] = result
test_df['Actual'] = y_true
test_df['Predicted'] = y_pred
print(test_df.head(500))
# test_df.drop(columns=[""], inplace=True)
test_df.to_csv('outputdf.csv')
                
print("tp is: ",tp)
print("tn is: ",tn)
print("fp is: ",fp)
print("fn is: ",fn)

#compute accuracy and f1 score
Accuracy = (tp+tn)/(tp+tn+fp+fn)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F1_Score = (2*Recall*Precision)/(Precision + Recall)

print("Accuracy:", Accuracy)
print("Precision:", Precision)
print("Recall:", Recall)
print("F1 score:", F1_Score)