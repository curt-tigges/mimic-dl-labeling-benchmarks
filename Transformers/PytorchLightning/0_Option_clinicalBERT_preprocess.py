import pandas as pd

from clinicalBERT_common import *

train_df = pd.read_csv('%s/train_50.csv' % MIMIC_3_DIR)
train_df['LABELS'] = train_df['LABELS'].apply(lambda x: x.split(';'))
train_df.to_pickle("./train.pkl")

eval_df = pd.read_csv('%s/dev_50.csv' % MIMIC_3_DIR)
eval_df['LABELS'] = eval_df['LABELS'].apply(lambda x: x.split(';'))
eval_df.to_pickle("./dev.pkl")

test_df = pd.read_csv('%s/test_50.csv' % MIMIC_3_DIR)
test_df['LABELS'] = test_df['LABELS'].apply(lambda x: x.split(';'))
test_df.to_pickle("./test.pkl")
