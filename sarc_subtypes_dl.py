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
df_no_dup = df.drop_duplicates(subset=['Hugo_Symbol'], keep='first')

print(df_no_dup)

# %%
