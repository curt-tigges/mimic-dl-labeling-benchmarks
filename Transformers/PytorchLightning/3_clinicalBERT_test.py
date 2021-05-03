import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from cs_utils import *
from mimic_constants import *
from mimic_evaluation import *
from mimic_models import *
from mimic_utils import *

############################################
# Prepare testing
############################################
with open(BEST_MODEL_INFO_PATH, 'r') as f:
    best_model_path = f.readline().strip()
    logger.info('Best model checkpoint path: {}'.format(best_model_path.strip()))

with open(BEST_THRESHOLD_INFO_PATH, 'r') as f:
    opt_thresh = float(f.readline().strip())
    logger.info('Optimal threshold: {}'.format(opt_thresh))

# Put model in evaluation mode
logger.info('Load trained model start')
model = MimicClassifier.load_from_checkpoint(checkpoint_path=best_model_path)
model = model.to(device)
model.eval()
logger.info('Load trained model end')

test_df = load_pickle(TEST_PICKLE)
x_test = test_df['TEXT']

mlb = MultiLabelBinarizer()
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

input_ids = load_pickle(PICKLE_TEST_INPUT_IDS)
attention_masks = load_pickle(PICKLE_TEST_ATTENTION_MASKS)

############################################
# 6. Metrics Report (Top 100)
############################################
# Prediction on test set
flat_pred_outs = 0
flat_true_labels = 0

# Tracking variables
pred_outs = load_pickle(TEST_PREDICT_OUTS)
true_labels = load_pickle(TEST_TRUE_LABELS)

# Combine the results across all batches.
flat_pred_outs = np.concatenate(pred_outs, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# convert labels to 1D array
y_true = flat_true_labels.ravel()

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


