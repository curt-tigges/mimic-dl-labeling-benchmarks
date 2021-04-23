import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import torch.nn as nn

import time
start_time = time.time()

def func_log(msg):
    cur = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - start_time
    print("{} ({:.3f}): {}".format(cur, elapsed, msg))

func_log("start")


# %%

# Setting up the device for GPU usage

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# %% md

## Load data

# %%

# change to where you store mimic3 data
MIMIC_3_DIR = '/CS598-DLH/caml-mimic/mimicdata/mimic3'

train_df = pd.read_csv('%s/train_50.csv' % MIMIC_3_DIR)

train_df.head()


# %% md

## Preprocess Data

# %%

# split labels by ";", then convert to list
def split_lab(x):
    # print(x)
    return x.split(";")


train_df['LABELS'] = train_df['LABELS'].apply(split_lab)

train_df.head()

# %%

# check top 50 code
top_50 = pd.read_csv('%s/TOP_50_CODES.csv' % MIMIC_3_DIR)

top_50.head().values

# %%

# load multi label binarizer for one-hot encoding
mlb = MultiLabelBinarizer(sparse_output=True)

# labels_onehot = mlb.fit_transform(train_df.pop('LABELS'))
# labels_onehot[0][1]

# %%

# change label to one-hot encoding per code
train_df = train_df.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(train_df.pop('LABELS')),
        index=train_df.index,
        columns=mlb.classes_))

train_df.head()

# %%

# Convert columns to list of one hot encoding
icd_classes_50 = mlb.classes_

train_df['labels'] = train_df[icd_classes_50].values.tolist()

train_df.head()

# %%

# check if one-hot encoding is correct
len(train_df.labels[0])

# %%

# convert into 2 columns dataframe
train_df = pd.DataFrame(train_df, columns=['TEXT', 'labels'])
train_df.columns = ['text', 'labels']
train_df.head()

# %% md

### Prepare Eval data

# %%

# same as train data preparation, but for evaluation
eval_df = pd.read_csv('%s/dev_50.csv' % MIMIC_3_DIR)

eval_df['LABELS'] = eval_df['LABELS'].apply(split_lab)

eval_df = eval_df.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(eval_df.pop('LABELS')),
        index=eval_df.index,
        columns=icd_classes_50))

eval_df['labels'] = eval_df[icd_classes_50].values.tolist()
eval_df = pd.DataFrame(eval_df, columns=['TEXT', 'labels'])
eval_df.columns = ['text', 'labels']

print(len(eval_df.labels[0]))
eval_df.describe

# %%

# same as train data preparation, but for evaluation
test_df = pd.read_csv('%s/test_50.csv' % MIMIC_3_DIR)

test_df['LABELS'] = test_df['LABELS'].apply(split_lab)

test_df = test_df.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(test_df.pop('LABELS')),
        index=test_df.index,
        columns=icd_classes_50))

test_df['labels'] = test_df[icd_classes_50].values.tolist()
test_df = pd.DataFrame(test_df, columns=['TEXT', 'labels'])
test_df.columns = ['text', 'labels']

print(len(test_df.labels[0]))
test_df.describe

# %% md

### Set Model Parameters

# %%

# Defining some key variables to configure model training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-05

# set tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")


# %% md

### Preparing Dataloader

# %%

# custom dataset for BERT class
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        '''
            set text as training data
            set labels as targets
        '''
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


# %%

# load df to dataset

print("TRAIN Dataset: {}".format(train_df.shape))
print("EVAL Dataset: {}".format(eval_df.shape))
print("TEST Dataset: {}".format(test_df.shape))

training_set = CustomDataset(train_df, tokenizer, MAX_LEN)
evaluation_set = CustomDataset(eval_df, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)

# %%

# data loader
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

eval_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

training_loader = DataLoader(training_set, **train_params)
evaluation_loader = DataLoader(evaluation_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# %% md

### Create model class from pretrained model

# %%

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        '''
            Load Pretrained model here
            Use return_dict=False for compatibility for 4.x

        '''
        self.l1 = transformers.AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT",
                                                         return_dict=False)
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)

        self.l2 = torch.nn.Dropout(0.3)

        '''
            Changed Linear Output layer to 50 based on the class
        '''
        self.l3 = torch.nn.Linear(768, 50)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = BERTClass()
model.to(device)
model = torch.nn.DataParallel(model)

# %%

# loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# %%

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# %% md

### Train fine-tuning model

# %%

def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            func_log(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %%

for epoch in tqdm(range(EPOCHS)):
    train(epoch)


# %% md

### Model Evaluation

# %%

# Evaluate the model

def validation(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


# %%

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    precision_score_micro = metrics.precision_score(targets, outputs, average='micro')
    precision_score_macro = metrics.precision_score(targets, outputs, average='macro')
    recall_score_micro = metrics.recall_score(targets, outputs, average='micro')
    recall_score_macro = metrics.recall_score(targets, outputs, average='macro')
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    auc_score_micro = metrics.roc_auc_score(targets, outputs, average='micro')
    auc_score_macro = metrics.roc_auc_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"Precision Score (Micro) = {precision_score_micro}")
    print(f"Precision Score (Macro) = {precision_score_macro}")
    print(f"Recall Score (Micro) = {recall_score_micro}")
    print(f"Recall Score (Macro) = {recall_score_macro}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"AUC Score (Micro) = {auc_score_micro}")
    print(f"AUC Score (Macro) = {auc_score_macro}")

    # %%

    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    precision_score_micro = metrics.precision_score(targets, outputs, average='micro')
    precision_score_macro = metrics.precision_score(targets, outputs, average='macro')
    recall_score_micro = metrics.recall_score(targets, outputs, average='micro')
    recall_score_macro = metrics.recall_score(targets, outputs, average='macro')
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    auc_score_micro = metrics.roc_auc_score(targets, outputs, average='micro')
    auc_score_macro = metrics.roc_auc_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"Precision Score (Micro) = {precision_score_micro}")
    print(f"Precision Score (Macro) = {precision_score_macro}")
    print(f"Recall Score (Micro) = {recall_score_micro}")
    print(f"Recall Score (Macro) = {recall_score_macro}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"AUC Score (Micro) = {auc_score_micro}")
    print(f"AUC Score (Macro) = {auc_score_macro}")

# %%

torch.save(model.state_dict(), "bluebert_state_dict_model.pt")

# %%

torch.save(model, "bluebert_model.pt")

# %%

func_log("end")