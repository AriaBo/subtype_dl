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


print("truth label")
print(truth_label)



y_binarized = label_binarize(truth_coded.CANCER_TYPE_CODE, classes=[0, 1, 2, 3, 4])
truth_coded = truth_coded.values
# Flatten truth_coded properly
truth_coded = truth_coded.flatten()


print("truth coded")
print(truth_coded) #y

print("y binarized")
print(y_binarized)

print(rna.values)

print("types!")
print(type(rna.values))
print(type(truth_coded))
print(type(y_binarized))


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(rna.values, truth_coded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) #validation data
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
#validation


# Define batch size
batch_size = 122

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#validation
# %%
#################################
#DEFINE THE NETWORK MODEL
##################################

class MultiClassModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassModel, self).__init__()
       
        hidden_size1 = 128
        hidden_size2 = 64
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.LayerNorm(hidden_size2),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(hidden_size2,num_classes)
        ) 

    def forward(self, x):
        x = self.mlp(x)
        return x

# Define input size and number of classes
input_size = X_train.shape[1]
num_classes = len(np.unique(truth_coded))

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiClassModel(input_size, num_classes).to(device)


# %%
###############################
#DEFINE LOSS FUNCTION AND OPTIMIZER 
################################

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# %%
###############################
#TRAINING AND EVALUATION LOOP WITH ROC CURVE
###############################

# Training loop
epochs = 100

# Lists to store results
epoch_count = []
train_loss_values = []
test_loss_values = []

best_test_loss=np.inf

# Training and evaluation loop
for epoch in range(epochs):

    # Training mode
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == targets).sum().item()
        total_train += targets.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * correct_train / total_train

    # Evaluation mode
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == targets).sum().item()
            total_test += targets.size(0)
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100.0 * correct_test / total_test
    
        # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:4} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Validation Loss: {test_loss:.4f} | Validation Accuracy: {test_accuracy:.2f}%')

    
    if test_loss < best_test_loss:
        best_test_loss =test_loss
        best_model_weights=model.state_dict()    
  

    # Append values to lists for plotting
    epoch_count.append(epoch)
    train_loss_values.append(train_loss)
    test_loss_values.append(test_loss)

model.load_state_dict(best_model_weights)
#deploy on test set 
#test loader + dataset
#no epoch but yust one 

#ROC and CONF MATRIX on this test set data nna3



#%%
#############################
# PLOT ROC
##############################

# ROC Curve Calculation and Plotting
all_targets = np.array(all_targets)
all_outputs = np.array(all_outputs)
all_outputs_prob = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], all_outputs_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'darkred']

cancer_type_mapping = {
    0: 'Dedifferentiated Liposarcoma',
    1: 'Leiomyosarcoma',
    2: 'Myxofibrosarcoma',
    3: 'Undifferentiated Pleomorphic Sarcoma',
    4: 'Synovial Sarcoma'
}

for i, color in zip(range(len(cancer_type_mapping)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(cancer_type_mapping[i], roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic to Multi-Class Sarc Subtypes')
plt.legend(loc="lower right")
plt.savefig('/home/arianna/subtype_dl/roc_curve_multiclass.pdf', format='pdf', dpi=300)
plt.show()


# %%
######################################
#CALCULATE ACCURACY AND PLOT CONFUSIUON MATRIX
######################################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import torch

# Assuming you have:
# - all_targets: list of true labels (0, 1, 2, 3, 4) for each subtype
# - all_outputs: list of raw logits from your model (before softmax)
# - y_test_binarized: binarized labels obtained from label_binarize

# Convert outputs to probabilities
all_outputs = np.array(all_outputs)
all_outputs_prob = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()

# Get predicted class labels
y_pred = np.argmax(all_outputs_prob, axis=1)

print(y_pred)


#---------------
#ACCURACY

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

#------------------
#CONFUSION MATRIX

# Plot the confusion matrix 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Class labels (replace with your actual class names)
class_names = ['Dedifferentiated Liposarcoma', 'Leiomyosarcoma',  'Myxofibrosarcoma',
           'Undifferentiated Pleomorphic Sarcoma', 'Synovial Sarcoma']

# Plot the confusion matrix using seaborn with custom settings
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.set_theme(font_scale=1.2)  # Adjust font size for readability
cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
cmd.plot(cmap='Blues', values_format='d')  # Use 'Blues' colormap and integer format for values
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.savefig('/home/arianna/subtype_dl/conf_matrix_multiclass.pdf', format='pdf', dpi=300)
plt.show()

# %%
