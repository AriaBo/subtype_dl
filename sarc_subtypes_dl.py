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
#########################
#READ DATA
#########################
df=pd.read_csv("/home/arianna/subtype_dl/data_mrna_seq_v2_rsem.txt", sep='\t')
df = df.drop(df.columns[1], axis=1)

df_filtered = df.dropna(subset=['Hugo_Symbol'])

# Drop duplicates, keeping only the first occurrence
df_no_dup = df_filtered.drop_duplicates(subset=['Hugo_Symbol'], keep='first')

df_T=df_no_dup.T
df_T.iloc[0, 0]="SAMPLE_ID"

print(df_T)

# %%
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
#READ MRNA DATA AND PREPROCESS THEM 
#####################################

rna=pd.read_csv( "/mnt/bulk/arianna/ATLAS_exp/sarc_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt", sep='\t')

#traspose + keep only gene symbol
#-----------------------------

#remove the column with entrez_gene_id
rna = rna.drop(rna.columns[1], axis=1) 
        
#traspose the rna dataframe 
rna=rna.T
        
#use the first row (with gene names as the column names for the dataframe)
new_colnames = rna.iloc[0, :]

# Assign the new column names (dropping the first row to avoid duplicates)
rna.columns = new_colnames
rna=rna[1:]  # Return the DataFrame excluding the first row (used for column names)

#remove columns with no gene name 
rna = rna[rna.columns.dropna()]


#remove missing genes
#-------------------------------
 
# Calculate the sum of each column (excluding the first two columns)
column_sums = rna.sum()

# Filter out columns where the sum is less than 5
filtered_columns = column_sums[column_sums >= 5].index
mask = ~column_sums.index.isin(filtered_columns)
remaining_columns = column_sums.index[mask]
print(rna[remaining_columns])
        
# Keep only the columns where the sum is greater than or equal to 5
rna = rna[filtered_columns]

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