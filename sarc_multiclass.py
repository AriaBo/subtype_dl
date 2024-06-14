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

#%%
X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, n_informative=15, random_state=42)

print(X)

print(y)
# %%

#import rna data 
truth=pd.read_csv("/home/arianna/subtype_dl/truth_encoded.csv", sep='\t')
rna=pd.read_csv("/home/arianna/subtype_dl/rna_filtered.csv", sep='\t', index_col=0)
truth_label=pd.read_csv("/home/arianna/subtype_dl/truth_label", sep='\t')

print(truth)
print(truth_label)
print(rna.values)

print(truth_label['CANCER_TYPE_DETAILED'].unique())

# Create a mapping of cancer types to numbers
cancer_type_mapping = {'Dedifferentiated Liposarcoma': 0,  
                      'Leiomyosarcoma': 1,
                      'Myxofibrosarcoma' : 2,
                      'Undifferentiated Pleomorphic Sarcoma' :3,
                      'Synovial Sarcoma': 4}


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(rna.values, truth_label, test_size=0.2, random_state=42)
y_train_binarized, y_test_binarized = train_test_split(truth.values, test_size=0.2, random_state=42)

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
