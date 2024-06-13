#%%
import torch 
from torch import nn

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# Settings for matplotlib
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Specify float format for pandas tables
pd.options.display.float_format = '{:.3f}'.format


#%%
#####################################
#GROUNDTRUTH
#####################################

#copy the groundtruth from data on github (analysis_data)
sarc_truth=pd.read_csv("/home/arianna/subtype_dl/data_clinical_sample.txt", sep='\t', skiprows=4)
sarc_truth=sarc_truth.loc[:, ["PATIENT_ID", "SAMPLE_ID", "CANCER_TYPE_DETAILED"]]

#remove cancer subtypes with < 10 samples 
sarc_truth = sarc_truth[~sarc_truth['CANCER_TYPE_DETAILED'].isin(['Desmoid/Aggressive Fibromatosis', 'Malignant Peripheral Nerve Sheath Tumor'])]
# Replace values in the "CANCER_TYPE_DETAILED" column
sarc_truth['CANCER_TYPE_DETAILED'] = sarc_truth['CANCER_TYPE_DETAILED'].replace(
    'Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma', 
    'Undifferentiated Pleomorphic Sarcoma'
)
sarc_truth.drop(columns=['PATIENT_ID'], inplace=True)
sarc_truth.set_index('SAMPLE_ID', inplace=True)

print(sarc_truth.loc[:, "CANCER_TYPE_DETAILED"].unique())
print(sarc_truth)

# %%
#####################################
#READ MRNA DATA AND PREPROCESS  
#####################################

rna=pd.read_csv( "/home/arianna/subtype_dl/data_mrna_seq_v2_rsem.txt", sep='\t')

#traspose + keep only gene symbol
#-----------------------------

#remove the column with entrez_gene_id
rna = rna.drop(rna.columns[1], axis=1) 

rna_filtered = rna.dropna(subset=['Hugo_Symbol'])

# Drop duplicates (in Hugo_Symbol), keeping only the first occurrence
rna_no_dup = rna_filtered.drop_duplicates(subset=['Hugo_Symbol'], keep='first')

#traspose the rna dataframe 
rna=rna_no_dup.T
      
#use the first row (with gene names as the column names for the dataframe)
new_colnames = rna.iloc[0, :]

# Assign the new column names (dropping the first row to avoid duplicates)
rna.columns = new_colnames
rna=rna[1:]  # Return the DataFrame excluding the first row (used for column names)

print(rna)

#%%
#################################
#FILTER SARC_TRUTH AND RNA TO HAVE THE SAME SAMPLE 
##############################

rna.rename_axis('SAMPLE_ID', inplace=True)

# Get the intersection of sample IDs between sarc_truth and rna
common_sample_ids = sarc_truth.index.intersection(rna.index)

# Filter sarc_truth to keep only the common sample IDs
sarc_truth_filtered = sarc_truth.loc[common_sample_ids]

# Filter rna to keep only the common sample IDs
rna_filtered = rna.loc[common_sample_ids]

print(sarc_truth_filtered)
print(rna_filtered)

# %%
#######################################
#CONVERT THE LABELS TO A DUMMY VARIABLE
#######################################

print(rna_filtered.info())
sarc_truth_filtered.describe(include=object)

# One-hot encoding of labels
truth_encoded = pd.get_dummies(sarc_truth_filtered, columns=['CANCER_TYPE_DETAILED'])  

# Result
print(truth_encoded)

# %%
####################################
#CONVERT TO TESOR + SPLIT TRAIN AND TEST DATASET
####################################

x_data = rna_filtered.values.astype(np.float32)  
x_data_tensor = torch.from_numpy(x_data).to(torch.float32)

y_data = truth_encoded.values.astype(np.float32) 
y_data_tensor = torch.from_numpy(y_data).to(torch.float32)

print(type(x_data_tensor))
print(type(y_data_tensor))

#%%

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, shuffle=True)

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)



# %%
##############################
#USE THE GPU
##############################

device = 'cuda' if torch.cuda.is_available() else print ('error')
print(device)

# %%
##########################
#INITIALISE THE MODEL 
##########################

class SubtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(19, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 96)
        self.linear4 = nn.Linear(96, 32)
        self.linear5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        return self.linear5(self.relu(self.linear4(self.dropout(self.relu(self.linear3(self.dropout(self.relu(self.linear2(self.dropout(self.relu(self.linear1(x))))))))))))
