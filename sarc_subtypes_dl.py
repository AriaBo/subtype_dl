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

sarc_truth_filtered.to_csv("/home/arianna/subtype_dl/truth_label", sep='\t')
truth_encoded.to_csv("/home/arianna/subtype_dl/truth_encoded.csv", sep='\t')
rna_filtered.to_csv("/home/arianna/subtype_dl/rna_filtered.csv", sep='\t')

# %%
####################################
#CONVERT TO TESOR + SPLIT TRAIN AND TEST DATASET
####################################

x_data = rna_filtered.values.astype(np.float32)  
x_data_tensor = torch.from_numpy(x_data).to(torch.float32)

y_data = truth_encoded.values.astype(np.float32) 
y_data_tensor = torch.from_numpy(y_data).to(torch.float32)

print(x_data)
print(type(x_data_tensor))
print(type(y_data_tensor))

#%%

x_train, x_test, y_train, y_test = train_test_split(x_data_tensor, y_data_tensor, test_size=0.30, shuffle=True)

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
        super(SubtypeModel, self).__init__()
        self.linear1 = nn.Linear(20511, 64)  # Adjusted input size to match x_train
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 96)
        self.linear4 = nn.Linear(96, 32)
        self.linear5 = nn.Linear(32, 5)  # Adjusted output to match the number of classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
  
    def forward(self, x):
        return self.linear5(self.relu(self.linear4(self.dropout(self.relu(self.linear3(self.dropout(self.relu(self.linear2(self.dropout(self.relu(self.linear1(x))))))))))))

#istantiate the model and move it to gpu

model = SubtypeModel().to(device)
model

#define hyperparameters 
learning_rate = 0.003
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# Forward pass
logits = model(x_train)
print('logits:', logits[:5])

# Logits -> Probabilities b/n 0 and 1 -> Rounded to 0 or 1
pred_probab = torch.round(torch.sigmoid(logits))
print('probabilities:', pred_probab[0:5])


#define accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc

# %%
###############################
#TRAIN THE MODEL
###############################
# Number of epochs
epochs = 100

# Send data to the device
x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Empty loss lists to track values
epoch_count, train_loss_values, test_loss_values = [], [], []

# Loop through the data
for epoch in range(epochs):

    # Put the model in training mode
    model.train()

    y_logits = model(x_train).squeeze() # forward pass to get predictions; squeeze the logits into the same shape as the labels
    y_pred = torch.round(torch.sigmoid(y_logits)) # convert logits into prediction probabilities

    loss = loss_fn(y_logits, y_train) # compute the loss   
    acc = accuracy_fn(y_train.int(), y_pred) # calculate the accuracy; convert the labels to integers

    optimizer.zero_grad() # reset the gradients so they don't accumulate each iteration
    loss.backward() # backward pass: backpropagate the prediction loss
    optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
    
    # Put the model in evaluation mode
    model.eval() 

    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))    

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test.int(), test_pred)    
    
    # Print progress a total of 20 times
    if epoch % int(epochs / 20) == 0:
        print(f'Epoch: {epoch:4.0f} | Train Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Accuracy: {test_acc:.2f}%')

        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().cpu().numpy())
        test_loss_values.append(test_loss.detach().cpu().numpy())

#%%
####################################
#PLOT
####################################

plt.plot(epoch_count, train_loss_values, label='Training Loss')
plt.plot(epoch_count, test_loss_values, label='Test Loss')
plt.title('Training & Test Loss Curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()



# %%
