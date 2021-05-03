from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

from cs_utils import *
from mimic_constants import *
from mimic_models import *
from mimic_utils import *

with open(BEST_MODEL_INFO_PATH, 'r') as f:
    best_model_path = f.readline().strip()
    logger.info('Best model checkpoint path: {}'.format(best_model_path.strip()))

############################################
# 5. Find best threshold
############################################
logger.info("Best threshold pick start")

# Tokenize all texts in x_test
input_ids = load_pickle(PICKLE_TEST_INPUT_IDS)
attention_masks = load_pickle(PICKLE_TEST_ATTENTION_MASKS)

train_df = load_pickle(TRAIN_PICKLE)
test_df = load_pickle(TEST_PICKLE)

x_tr = train_df['TEXT']
x_test = test_df['TEXT']

mlb = MultiLabelBinarizer()
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

logger.info(mlb.classes_)
logger.info('label encoding end')

logger.info('BERT tokenizer load start')
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
logger.info('BERT tokenizer load end')

# for small test
if FOR_LOCAL_TEST:
    x_test = x_test[0:2]
    y_test = y_test[0:2]


logger.info('Setup test dataset for BERT start')
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
logger.info('Load trained model start')
model = MimicClassifier.load_from_checkpoint(checkpoint_path=best_model_path)
model = model.to(device)
model.eval()
logger.info('Load trained model end')

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

with open(TEST_PREDICT_OUTS, 'w') as f:
    pickle.dump(flat_pred_outs, f)

with open(TEST_TRUE_LABELS, 'w') as f:
    pickle.dump(flat_true_labels, f)

# Predictions of Tags in Test set
# define candidate threshold values
threshold = np.arange(0.2, 0.8, 0.01)

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
opt_thresh = round(opt_thresh, 2)  # 0.22000000000000003 => 0.22
logger.info(f'Optimal Threshold Value = {opt_thresh}')

with open(BEST_THRESHOLD_INFO_PATH, 'w') as f:
    f.write(str(opt_thresh))
