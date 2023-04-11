import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
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

# Load the clinical_bio_bert model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# model = AutoModelForSequenceClassification.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")
# tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBert-PubMed200kRCT")

#Data size taken to train and test the model
data_size = 20000

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
    
# Load the dataset of eligible patients and ineligible patients
# and create a list of embeddings for each patient
patients = df
embeddings = []
labels = []
for i in range(len(patients)):
    tokens = tokenizer(patients.iloc[i,1], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    embeddings.append(embedding)
    labels.append(patients.iloc[i,0])

embeddings = np.squeeze(embeddings)

# Split the embeddings and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

unique_labels = np.unique(labels)
# Train an SVM classifier on the training set
clf = SVC(kernel="linear", random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Make a prediction on a new patient
new_patient = " recurrent childhood lymphoblastic lymphoma diagnosis and bone marrow may be used in conjunction with blood progenitor cells."
tokens = tokenizer(new_patient, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    embedding = model(**tokens).last_hidden_state.mean(dim=1).numpy()
embedding = np.squeeze(embedding)
prediction = clf.predict([embedding])

#Metrics to compute f1 score
#true positives, true negatives, false positives, false negatives
tp=0
tn=0
fp=0
fn=0

for i in range(len(y_test)):
        if(y_test[i]==y_pred[i]):
            if(y_test[i]==1):
                tp=tp+1
            else:
                tn=tn+1
        else:
            if(y_test[i]==1):
                fn=fn+1
            else:
                fp=fp+1
                
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