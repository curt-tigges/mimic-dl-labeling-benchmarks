import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from cs_utils import *
from mimic_constants import *
from mimic_evaluation import *
from mimic_utils import *

with open(BEST_THRESHOLD_INFO_PATH, 'w') as f:
    opt_thresh = float(f.read())

test_df = load_pickle('{}/test.pkl'.format(MIMIC_3_DIR))
x_test = test_df['TEXT']

mlb = MultiLabelBinarizer()
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

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


