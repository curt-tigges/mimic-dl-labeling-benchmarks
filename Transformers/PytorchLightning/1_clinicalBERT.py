import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import SequentialSampler
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from clinicalBERT_common import *
from cs_utils import *
from evaluation import *


############################################
# 1. Load & Pre-process the data
############################################
train_df = load_pickle('{}/train.pkl'.format(MIMIC_3_DIR))
eval_df = load_pickle('{}/dev.pkl'.format(MIMIC_3_DIR))
test_df = load_pickle('{}/test.pkl'.format(MIMIC_3_DIR))

x_tr = train_df['TEXT']
x_val = eval_df['TEXT']
x_test = test_df['TEXT']

logger.info('load data end')


############################################
# 2.1 Encoding label
############################################
mlb = MultiLabelBinarizer()
y_tr = mlb.fit_transform(train_df['LABELS'])  # y tag(label)
y_val = mlb.fit_transform(eval_df['LABELS'])  # y tag(label)
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

logger.info(mlb.classes_)
logger.info('label encoding end')


############################################
# 2.2 Check the length of Clinical Notes
############################################
# word_cnt = train_df.pop('length')
# plt.figure(figsize=[8, 5])
# plt.hist(word_cnt, bins=40)
# plt.xlabel('Word Count/Clinical Note')
# plt.ylabel('# of Occurences')
# plt.title("Frequency of Word Counts/Clinical Note")
# plt.show()


############################################
# 3. Define our model
############################################
# Initialize the Bert tokenizer
logger.info('BERT tokenizer load start')
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
logger.info('BERT tokenizer load end')

# Instantiate and set up the data_module
data_module = MimicDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer, BATCH_SIZE, MAX_LEN)
data_module.setup()


############################################
# 4. Train the model
############################################
# Instantiate the classifier model
steps_per_epoch = len(x_tr) // BATCH_SIZE
model = MimicClassifier(n_classes=50, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

# Initialize Pytorch Lightning callback for Model checkpointing
# saves a file like: input/MIMIC-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # monitored quantity
    filename='MIMIC-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,  # save the top 3 models
    mode='min',  # mode of the monitored quantity for optimization
)

# Instantiate the Model Trainer
if torch.cuda.is_available():
    # for GPUs
    trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, accelerator='ddp',
                         callbacks=[checkpoint_callback], progress_bar_refresh_rate=30)
else:
    # for CPUs
    trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=0,
                         callbacks=[checkpoint_callback], progress_bar_refresh_rate=30)

# Train the Classifier Model
trainer.fit(model, data_module)

logger.info("train end")

# Retrieve the checkpoint path for best model
model_path = checkpoint_callback.best_model_path
logger.info('Best model path: {}'.format(model_path))


############################################
# 5. Find best threshold
############################################
logger.info("Best threshold pick start")

# Tokenize all questions in x_test
input_ids = []
attention_masks = []

train_df = load_pickle('{}/train.pkl'.format(MIMIC_3_DIR))
eval_df = load_pickle('{}/dev.pkl'.format(MIMIC_3_DIR))
test_df = load_pickle('{}/test.pkl'.format(MIMIC_3_DIR))

x_tr = train_df['TEXT']
x_test = test_df['TEXT']

mlb = MultiLabelBinarizer()
y_tr = mlb.fit_transform(train_df['LABELS'])  # y tag(label)
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

# for small test
if FOR_LOCAL_TEST:
    x_test = x_test[0:2]
    y_test = y_test[0:2]

input_ids = []
attention_masks = []

logger.info('Setup test dataset for BERT start')
for quest in x_test:
    encoded_quest = Bert_tokenizer.encode_plus(
        quest,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    # Add the input_ids from encoded question to the list.
    input_ids.append(encoded_quest['input_ids'])
    # Add its attention mask
    attention_masks.append(encoded_quest['attention_mask'])

# Now convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_test)

logger.info('Setup test dataset for BERT end')

# Set the batch size.
TEST_BATCH_SIZE = 64

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE)

# Prediction on test set
flat_pred_outs = 0
flat_true_labels = 0

# Put model in evaluation mode
model.eval()

# Tracking variables
pred_outs, true_labels = [], []

# Predict
for batch in pred_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_attn_mask, b_labels = batch

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        pred_out = model(b_input_ids, b_attn_mask)
        pred_out = torch.sigmoid(pred_out)
        # Move predicted output and labels to CPU
        pred_out = pred_out.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    pred_outs.append(pred_out)
    true_labels.append(label_ids)

# Combine the results across all batches.
flat_pred_outs = np.concatenate(pred_outs, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Predictions of Tags in Test set
# define candidate threshold values
threshold = np.arange(0.2, 0.3, 0.01)


# convert probabilities into 0 or 1 based on a threshold value
def classify(pred_prob, thresh):
    y_pred = []

    for tag_label_row in pred_prob:
        temp = []
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1)  # Infer tag value as 1 (present)
            else:
                temp.append(0)  # Infer tag value as 0 (absent)
        y_pred.append(temp)

    return y_pred


from sklearn import metrics
scores = []  # Store the list of f1 scores for prediction on each threshold

# convert labels to 1D array
y_true = flat_true_labels.ravel()

for thresh in threshold:
    # classes for each threshold
    pred_bin_label = classify(flat_pred_outs, thresh)

    # convert to 1D array
    y_pred = np.array(pred_bin_label).ravel()

    scores.append(metrics.f1_score(y_true, y_pred))

opt_thresh = threshold[scores.index(max(scores))]
logger.info(f'Optimal Threshold Value = {opt_thresh}')


############################################
# 6. Metrics Report (Top 100)
############################################
y_pred_labels = classify(flat_pred_outs, opt_thresh)
y_pred = np.array(y_pred_labels).ravel()  # Flatten

logger.info('\n' + metrics.classification_report(y_true, y_pred))

y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

df = pd.DataFrame({'Body': x_test, 'Actual Labels': y_act, 'Predicted Labels': y_pred})
print(df.sample(min(len(df), 100)))


############################################
# 6.1 Metric Report (Detailed)
############################################
y = flat_true_labels            # binary ground truth matrix
yhat = np.array(y_pred_labels)  # binary predictions matrix
yhat_raw = flat_pred_outs       # score matrix (floats)

metric = all_metrics(yhat, y, k=5, yhat_raw=yhat_raw)
for key, value in sorted(metric.items()):
    print('{}: {}'.format(key, value))

print('===')


