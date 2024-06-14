#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%

#import rna data 
truth=pd.read_csv("/home/arianna/subtype_dl/truth_encoded.csv", sep='\t')
rna=pd.read_csv("/home/arianna/subtype_dl/rna_filtered.csv", sep='\t', index_col=0)
truth_label=pd.read_csv("/home/arianna/subtype_dl/truth_label", sep='\t')

#print(truth_label['CANCER_TYPE_DETAILED'].unique())

# Create a mapping of cancer types to numbers
cancer_type_mapping = {'Dedifferentiated Liposarcoma': 0,  
                      'Leiomyosarcoma': 1,
                      'Myxofibrosarcoma' : 2,
                      'Undifferentiated Pleomorphic Sarcoma' :3,
                      'Synovial Sarcoma': 4}

# Apply the mapping to the 'CANCER_TYPE_DETAILED' column
truth_label['CANCER_TYPE_CODE'] = truth_label['CANCER_TYPE_DETAILED'].map(cancer_type_mapping)

# Create a new DataFrame with the same index but with the corresponding number
truth_coded = truth_label[['CANCER_TYPE_CODE']]

print(truth_coded)

y_binarized = label_binarize(truth_coded.CANCER_TYPE_CODE, classes=[0, 1, 2, 3, 4])
truth_coded= truth_coded.values

print(y_binarized)

print(rna.values)

print("types!")
print(type(rna.values))
print(type(truth_coded))
print(type(y_binarized))


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(rna.values, truth_coded, test_size=0.2, random_state=42)
y_train_binarized, y_test_binarized = train_test_split(y_binarized, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
x_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Convert to PyTorch tensors for ROC curve (binarized labels)
y_train_binarized = torch.tensor(y_train_binarized, dtype=torch.float32)
y_test_binarized = torch.tensor(y_test_binarized, dtype=torch.float32)

# Create TensorDataset
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Define batch size
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
#################################
#DEFINE THE NETWORK MODEL
##################################

class MultiClassModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 96)
        self.linear4 = nn.Linear(96, 32)
        self.linear5 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x))
        x = self.linear5(x)  # Output logits for each class
        return x

# Define input size and number of classes
input_size = X_train.shape[1]
num_classes = len(np.unique(y))

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiClassModel(input_size, num_classes).to(device)


# %%
###############################
#DEFINE LOSS FUNCTION AND OPTIMIZER 
################################

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
